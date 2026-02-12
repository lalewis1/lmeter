#!/usr/bin/env python3
"""
HTTPX-based load test utility that simulates 30 users making 5 requests each
to the endpoints specified in prez_endpoint.txt and fuseki_endpoint.txt.
For prez endpoint: uses URL-based search with varied search terms.
For fuseki endpoint: uses SPARQL queries with templated search terms (constructQuery.rq and countQuery.rq).
"""

import asyncio
import logging
import random
import string
from datetime import datetime
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from io import StringIO

import httpx
from jinja2 import Template
from rdflib import Graph


async def read_endpoint(filename):
    """Read the endpoint URL from a file"""
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        logging.error(f"Error: {filename} not found")
        return None
    except Exception as e:
        logging.error(f"Error reading {filename}: {e}")
        return None


async def read_query_templates():
    """Read the SPARQL query templates from constructQuery.rq, countQuery.rq, and anotQuery.rq"""
    try:
        with open("constructQuery.rq", "r") as f:
            construct_template = f.read()
        with open("countQuery.rq", "r") as f:
            count_template = f.read()
        with open("anotQuery.rq", "r") as f:
            anot_template = f.read()
        return construct_template, count_template, anot_template
    except FileNotFoundError as e:
        logging.error(f"Error: {e.filename} not found")
        return None, None, None
    except Exception as e:
        logging.error(f"Error reading query templates: {e}")
        return None, None, None


def setup_logging():
    """
    Setup logging to both console and timestamped log file
    """
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"load_test_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return log_filename


def modify_search_term(url, suffix_length=4):
    """
    Modify the q= query parameter with a completely random search term
    of variable length (4-8 characters) for URL-based endpoints
    """
    # Generate completely random search term of variable length (4-8 chars)
    term_length = random.randint(4, 8)
    random_search_term = "".join(
        random.choices(string.ascii_uppercase + string.digits, k=term_length)
    )

    # Parse the URL
    parsed = urlparse(url)
    query_params = parse_qs(parsed.query)

    # Update the q parameter with the random search term
    query_params["q"] = [random_search_term]

    # Rebuild the URL
    new_query = urlencode(query_params, doseq=True)
    new_parsed = parsed._replace(query=new_query)

    return urlunparse(new_parsed)


def generate_sparql_query(template_content, search_term):
    """
    Generate a SPARQL query by templating the search term
    """
    template = Template(template_content)
    return template.render(search_term=search_term)


def parse_construct_results(response_text):
    """
    Parse the CONSTRUCT query results to extract distinct IRIs
    Returns a set of unique IRIs found in subject, predicate, and object positions
    """
    try:
        # Parse the response as N-Triples
        graph = Graph()
        graph.parse(source=StringIO(response_text), format="nt")
        
        # Extract all unique IRIs from the graph
        iris = set()
        for subject, predicate, obj in graph:
            if subject and str(subject).startswith('http'):
                iris.add(str(subject))
            if predicate and str(predicate).startswith('http'):
                iris.add(str(predicate))
            if obj and str(obj).startswith('http'):
                iris.add(str(obj))
        
        return list(iris)
    except Exception as e:
        logging.error(f"Error parsing construct results: {e}")
        return []


def generate_annotation_query(template_content, iris):
    """
    Generate an annotation SPARQL query by templating the list of IRIs
    """
    template = Template(template_content)
    return template.render(iris=iris)


async def make_request(
    client, url, user_id, request_num, request_type="url", query_templates=None
):
    """Make a single HTTP request and return timing information"""
    if request_type == "url":
        modified_url = modify_search_term(url)
        request_url = modified_url
        print_url = modified_url
    elif request_type == "sparql":
        # Generate completely random search term of variable length (4-8 chars)
        term_length = random.randint(4, 8)
        search_term = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=term_length)
        )

        # Generate all SPARQL queries
        construct_template, count_template, anot_template = query_templates
        construct_query = generate_sparql_query(construct_template, search_term)
        count_query = generate_sparql_query(count_template, search_term)
        request_url = url
        print_url = f"{url} [search_term={search_term}]"
    else:
        raise ValueError(f"Unknown request type: {request_type}")

    try:
        start_time = asyncio.get_event_loop().time()

        if request_type == "sparql":
            # For SPARQL endpoint, send construct, count, and annotation queries sequentially
            headers = {
                "Content-Type": "application/sparql-query",
                "Accept": "application/n-triples",  # Request N-Triples for easier parsing
            }
            
            # Execute construct query
            response1 = await client.post(
                request_url, content=construct_query, headers=headers
            )
            
            # Parse construct results to get IRIs for annotation query
            iris = []
            if response1.status_code == 200:
                iris = parse_construct_results(response1.text)
            
            # Execute count query
            headers_count = {
                "Content-Type": "application/sparql-query",
                "Accept": "application/json",
            }
            response2 = await client.post(
                request_url, content=count_query, headers=headers_count
            )
            
            # Execute annotation query if we have IRIs
            response3 = None
            if iris:
                anot_query = generate_annotation_query(anot_template, iris)
                headers_anot = {
                    "Content-Type": "application/sparql-query",
                    "Accept": "application/n-triples",
                }
                response3 = await client.post(
                    request_url, content=anot_query, headers=headers_anot
                )
            
            # Use the combined duration and average status
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            
            # Calculate average status code
            status_codes = [response1.status_code, response2.status_code]
            if response3:
                status_codes.append(response3.status_code)
            status = sum(status_codes) // len(status_codes)

        else:
            # For URL endpoint, send GET request
            response = await client.get(request_url)
            end_time = asyncio.get_event_loop().time()
            duration = end_time - start_time
            status = response.status_code

        logging.info(
            f"User {user_id:2d} Request {request_num:2d}: {status} in {duration:.3f}s - {print_url}"
        )

        result = {
            "user_id": user_id,
            "request_num": request_num,
            "status": status,
            "duration": duration,
            "url": print_url,
        }
        
        # Add additional info for SPARQL requests
        if request_type == "sparql":
            result["iris_found"] = len(iris)
            result["annotation_query_executed"] = response3 is not None
        
        return result

    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - asyncio.get_event_loop().time()

        logging.error(f"User {user_id:2d} Request {request_num:2d}: ERROR - {str(e)}")

        return {
            "user_id": user_id,
            "request_num": request_num,
            "status": "ERROR",
            "duration": duration,
            "url": print_url,
            "error": str(e),
        }


async def user_simulation(
    user_id, base_url, requests_per_user, request_type="url", query_templates=None
):
    """Simulate a single user making multiple requests"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = []

        for request_num in range(1, requests_per_user + 1):
            result = await make_request(
                client, base_url, user_id, request_num, request_type, query_templates
            )
            results.append(result)

            # Add a small delay between requests from the same user
            await asyncio.sleep(0.1)

        return results


async def run_single_load_test(
    test_name, endpoint_file, request_type="url", num_users=30, requests_per_user=5
):
    """Run a single load test for a specific endpoint"""
    base_url = await read_endpoint(endpoint_file)

    if not base_url:
        print(f"Skipping {test_name} test - endpoint file not found or empty")
        return None

    query_templates = None
    if request_type == "sparql":
        query_templates = await read_query_templates()
        if not all(query_templates):
            print(f"Skipping {test_name} test - query templates not found")
            return None

    logging.info(f"{'='*80}")
    logging.info(f"{test_name.upper()} LOAD TEST")
    logging.info(f"{'='*80}")
    logging.info(
        f"Starting {test_name} load test with {num_users} users, {requests_per_user} requests each"
    )
    logging.info(f"Base URL: {base_url}")
    if request_type == "sparql":
        logging.info("Using constructQuery.rq, countQuery.rq, and anotQuery.rq templates")
    logging.info("-" * 80)

    # Create tasks for all users
    tasks = []
    for user_id in range(1, num_users + 1):
        task = asyncio.create_task(
            user_simulation(
                user_id, base_url, requests_per_user, request_type, query_templates
            )
        )
        tasks.append(task)

        # Stagger user start times slightly
        await asyncio.sleep(0.05)

    # Wait for all users to complete
    all_results = await asyncio.gather(*tasks)

    # Flatten results
    flat_results = [result for user_results in all_results for result in user_results]

    # Calculate statistics
    successful_requests = [r for r in flat_results if r["status"] != "ERROR"]
    failed_requests = [r for r in flat_results if r["status"] == "ERROR"]

    durations = [r["duration"] for r in successful_requests]

    logging.info("" + "=" * 80)
    logging.info(f"{test_name.upper()} LOAD TEST SUMMARY")
    logging.info("=" * 80)
    logging.info(f"Total requests: {len(flat_results)}")
    logging.info(f"Successful requests: {len(successful_requests)}")
    logging.info(f"Failed requests: {len(failed_requests)}")
    if request_type == "sparql":
        logging.info(f"Note: Each request executes 3 SPARQL queries (construct + count + annotation)")

    if durations:
        logging.info(f"Average response time: {sum(durations) / len(durations):.3f}s")
        logging.info(f"Minimum response time: {min(durations):.3f}s")
        logging.info(f"Maximum response time: {max(durations):.3f}s")

    if failed_requests:
        logging.info("Failed requests details:")
        for result in failed_requests:
            logging.info(
                f"  User {result['user_id']} Request {result['request_num']}: {result['error']}"
            )
    
    # Additional stats for SPARQL requests
    if request_type == "sparql":
        successful_sparql_requests = [r for r in successful_requests if r.get("iris_found") is not None]
        if successful_sparql_requests:
            total_iris = sum(r["iris_found"] for r in successful_sparql_requests)
            avg_iris = total_iris / len(successful_sparql_requests)
            annotation_queries = sum(1 for r in successful_sparql_requests if r.get("annotation_query_executed"))
            
            logging.info(f"SPARQL-specific statistics:")
            logging.info(f"  Average IRIs found per request: {avg_iris:.1f}")
            logging.info(f"  Annotation queries executed: {annotation_queries}/{len(successful_sparql_requests)}")

    return {
        "name": test_name,
        "total_requests": len(flat_results),
        "successful_requests": len(successful_requests),
        "failed_requests": len(failed_requests),
        "avg_response_time": sum(durations) / len(durations) if durations else 0,
        "min_response_time": min(durations) if durations else 0,
        "max_response_time": max(durations) if durations else 0,
    }


async def run_load_test():
    """Run the complete load test for both prez and fuseki endpoints"""
    num_users = 30
    requests_per_user = 5

    # Setup logging
    log_filename = setup_logging()
    logging.info("Starting comprehensive load test")
    logging.info("=" * 80)
    logging.info(f"Log file: {log_filename}")

    # Run prez endpoint test (URL-based)
    prez_results = await run_single_load_test(
        "Prez Endpoint",
        "prez_endpoint.txt",
        request_type="url",
        num_users=num_users,
        requests_per_user=requests_per_user,
    )

    # Run fuseki endpoint test (SPARQL-based)
    fuseki_results = await run_single_load_test(
        "Fuseki Endpoint",
        "fuseki_endpoint.txt",
        request_type="sparql",
        num_users=num_users,
        requests_per_user=requests_per_user,
    )

    # Print overall summary
    logging.info(f"{'='*80}")
    logging.info("OVERALL LOAD TEST SUMMARY")
    logging.info("=" * 80)

    if prez_results:
        logging.info(f"Prez Endpoint Results:")
        logging.info(f"  Total requests: {prez_results['total_requests']}")
        logging.info(f"  Successful: {prez_results['successful_requests']}")
        logging.info(f"  Failed: {prez_results['failed_requests']}")
        logging.info(f"  Avg response time: {prez_results['avg_response_time']:.3f}s")

    if fuseki_results:
        logging.info(f"Fuseki Endpoint Results:")
        logging.info(f"  Total requests: {fuseki_results['total_requests']}")
        logging.info(f"  Successful: {fuseki_results['successful_requests']}")
        logging.info(f"  Failed: {fuseki_results['failed_requests']}")
        logging.info(f"  Avg response time: {fuseki_results['avg_response_time']:.3f}s")

    logging.info(f"Load test completed!")
    logging.info(f"Results saved to: {log_filename}")


if __name__ == "__main__":
    asyncio.run(run_load_test())
