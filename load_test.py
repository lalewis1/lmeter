#!/usr/bin/env python3
"""
HTTPX-based load test utility that simulates 30 users making 5 requests each
to the endpoints specified in prez_endpoint.txt and fuseki_endpoint.txt.
For prez endpoint: uses URL-based search with varied search terms.
For fuseki endpoint: uses SPARQL queries with templated search terms.
"""

import asyncio
import random
import string
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import httpx
from jinja2 import Template


async def read_endpoint(filename):
    """Read the endpoint URL from a file"""
    try:
        with open(filename, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: {filename} not found")
        return None
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None


async def read_query_template():
    """Read the SPARQL query template from query.rq"""
    try:
        with open("query.rq", "r") as f:
            return f.read()
    except FileNotFoundError:
        print("Error: query.rq not found")
        return None
    except Exception as e:
        print(f"Error reading query.rq: {e}")
        return None


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


async def make_request(
    client, url, user_id, request_num, request_type="url", query_template=None
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

        # Generate SPARQL query
        sparql_query = generate_sparql_query(query_template, search_term)
        request_url = url
        print_url = f"{url} [search_term={search_term}]"
    else:
        raise ValueError(f"Unknown request type: {request_type}")

    try:
        start_time = asyncio.get_event_loop().time()

        if request_type == "sparql":
            # For SPARQL endpoint, send POST request with query
            headers = {
                "Content-Type": "application/sparql-query",
                "Accept": "application/json",
            }
            response = await client.post(
                request_url, content=sparql_query, headers=headers
            )
        else:
            # For URL endpoint, send GET request
            response = await client.get(request_url)

        end_time = asyncio.get_event_loop().time()

        duration = end_time - start_time
        status = response.status_code

        print(
            f"User {user_id:2d} Request {request_num:2d}: {status} in {duration:.3f}s - {print_url}"
        )

        return {
            "user_id": user_id,
            "request_num": request_num,
            "status": status,
            "duration": duration,
            "url": print_url,
        }

    except Exception as e:
        end_time = asyncio.get_event_loop().time()
        duration = end_time - asyncio.get_event_loop().time()

        print(f"User {user_id:2d} Request {request_num:2d}: ERROR - {str(e)}")

        return {
            "user_id": user_id,
            "request_num": request_num,
            "status": "ERROR",
            "duration": duration,
            "url": print_url,
            "error": str(e),
        }


async def user_simulation(
    user_id, base_url, requests_per_user, request_type="url", query_template=None
):
    """Simulate a single user making multiple requests"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        results = []

        for request_num in range(1, requests_per_user + 1):
            result = await make_request(
                client, base_url, user_id, request_num, request_type, query_template
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

    query_template = None
    if request_type == "sparql":
        query_template = await read_query_template()
        if not query_template:
            print(f"Skipping {test_name} test - query template not found")
            return None

    print(f"\n{'='*80}")
    print(f"{test_name.upper()} LOAD TEST")
    print(f"{'='*80}")
    print(
        f"Starting {test_name} load test with {num_users} users, {requests_per_user} requests each"
    )
    print(f"Base URL: {base_url}")
    print("-" * 80)

    # Create tasks for all users
    tasks = []
    for user_id in range(1, num_users + 1):
        task = asyncio.create_task(
            user_simulation(
                user_id, base_url, requests_per_user, request_type, query_template
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

    print("\n" + "=" * 80)
    print(f"{test_name.upper()} LOAD TEST SUMMARY")
    print("=" * 80)
    print(f"Total requests: {len(flat_results)}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {len(failed_requests)}")

    if durations:
        print(f"Average response time: {sum(durations) / len(durations):.3f}s")
        print(f"Minimum response time: {min(durations):.3f}s")
        print(f"Maximum response time: {max(durations):.3f}s")

    if failed_requests:
        print("\nFailed requests details:")
        for result in failed_requests:
            print(
                f"  User {result['user_id']} Request {result['request_num']}: {result['error']}"
            )

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

    print("Starting comprehensive load test")
    print("=" * 80)

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
    print(f"\n{'='*80}")
    print("OVERALL LOAD TEST SUMMARY")
    print("=" * 80)

    if prez_results:
        print(f"\nPrez Endpoint Results:")
        print(f"  Total requests: {prez_results['total_requests']}")
        print(f"  Successful: {prez_results['successful_requests']}")
        print(f"  Failed: {prez_results['failed_requests']}")
        print(f"  Avg response time: {prez_results['avg_response_time']:.3f}s")

    if fuseki_results:
        print(f"\nFuseki Endpoint Results:")
        print(f"  Total requests: {fuseki_results['total_requests']}")
        print(f"  Successful: {fuseki_results['successful_requests']}")
        print(f"  Failed: {fuseki_results['failed_requests']}")
        print(f"  Avg response time: {fuseki_results['avg_response_time']:.3f}s")

    print(f"\nLoad test completed!")


if __name__ == "__main__":
    asyncio.run(run_load_test())
