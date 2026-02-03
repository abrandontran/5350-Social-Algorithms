#!/usr/bin/env python3
"""
Bluesky API Helper Functions for Assignment 2: Analyzing Bluesky

This module provides helper functions for interacting with the Bluesky public API.
These handle the low-level API communication so you can focus on the analysis.

No authentication is required for read-only access to public data.

IMPORTANT NOTES:
-----------------------------
1. Rate Limiting: The API allows ~3,000 requests per 5 minutes. Add sleep(0.1)
   between requests to avoid being rate-limited. For large data collection,
   consider using longer delays (0.15-0.2s) to be safe.

2. Error Handling: The API can return errors for deleted accounts, private
   profiles, or temporary issues. Your code should handle these gracefully
   (e.g., skip the account, log the error, continue processing).

3. Data Collection Time: Collecting feeds from thousands of accounts takes
   significant time (30-60+ minutes). Consider:
   - Running collection scripts in the background
   - Saving intermediate results to resume if interrupted
   - Caching data to avoid re-collecting

4. API Limits: The getPostThread endpoint returns at most ~200 replies per post,
   biased toward EARLIER replies. This is NOT a uniformly random sample -- keep
   this in mind when analyzing reply data.

Dependencies: Only uses standard library (no pip install required)
"""

import json
import urllib.request
import urllib.error
import urllib.parse
import time
from datetime import datetime, timezone, timedelta

# Base URL for Bluesky public API
API_BASE = "https://public.api.bsky.app/xrpc"

# Default timeout for requests (seconds) - increase if you see timeout errors
DEFAULT_TIMEOUT = 15

# Rate limiting delay (seconds between requests)
# Increase to 0.15-0.2 if you get rate limit errors during large collection
RATE_LIMIT_DELAY = 0.1


def make_request(endpoint, params=None, timeout=DEFAULT_TIMEOUT):
    """
    Make a GET request to the Bluesky API.

    Args:
        endpoint: API endpoint (e.g., 'app.bsky.actor.getProfile')
        params: Dictionary of query parameters
        timeout: Request timeout in seconds

    Returns:
        Parsed JSON response as a dictionary, or None on error

    Common errors you may encounter:
        - 400: Bad request (invalid handle or URI format)
        - 404: Account not found (deleted or doesn't exist)
        - 429: Rate limited (slow down your requests)
        - 500/502/503: Server errors (retry after a delay)

    Tip: For robust data collection, consider wrapping calls in a retry loop
    with exponential backoff for transient errors.
    """
    url = f"{API_BASE}/{endpoint}"

    if params:
        query_string = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        url = f"{url}?{query_string}"

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        # Don't print for common "not found" errors during bulk collection
        if e.code not in [400, 404]:
            print(f"HTTP Error {e.code}: {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"URL Error: {e.reason}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        return None
    except TimeoutError:
        print(f"Timeout after {timeout}s")
        return None


# ============================================================================
# API Wrapper Functions
# These handle the API communication so you can focus on your analysis.
# ============================================================================

def get_profile(handle):
    """
    Get a user's profile information.

    Args:
        handle: Bluesky handle (e.g., 'senator.senate.gov')

    Returns:
        Dictionary with profile data, or None on error

    Example response fields:
        - 'did': Permanent identifier
        - 'handle': The handle
        - 'displayName': Display name
        - 'followersCount': Number of followers
        - 'followsCount': Number of accounts followed
        - 'postsCount': Number of posts
    """
    return make_request('app.bsky.actor.getProfile', {'actor': handle})


def get_follows(handle, limit=100, cursor=None):
    """
    Get accounts that a user follows (single page).

    Args:
        handle: Bluesky handle
        limit: Maximum results per request (max 100)
        cursor: Pagination cursor for subsequent requests

    Returns:
        Dictionary with 'follows' list and optional 'cursor' for pagination

    Note:
        To get all follows, call repeatedly with the returned cursor until
        no cursor is returned. See get_all_follows() for an example.
    """
    params = {'actor': handle, 'limit': min(limit, 100)}
    if cursor:
        params['cursor'] = cursor
    return make_request('app.bsky.graph.getFollows', params)


def get_all_follows(handle):
    """
    Get ALL accounts that a user follows (handles pagination automatically).

    Args:
        handle: Bluesky handle

    Returns:
        List of all followed accounts (each is a dict with 'handle', 'did',
        'displayName', etc.)

    Each followed account dict contains useful fields:
        - 'handle': The account's handle (e.g., 'user.bsky.social')
        - 'did': The account's permanent identifier
        - 'displayName': The account's display name (may be empty)
    """
    all_follows = []
    cursor = None

    while True:
        result = get_follows(handle, limit=100, cursor=cursor)
        if not result:
            break

        follows = result.get('follows', [])
        all_follows.extend(follows)

        cursor = result.get('cursor')
        if not cursor:
            break

        time.sleep(RATE_LIMIT_DELAY)

    return all_follows


def get_author_feed(handle, limit=50, cursor=None):
    """
    Get recent posts from a user's feed.

    Args:
        handle: Bluesky handle
        limit: Maximum posts to return (max 100)
        cursor: Pagination cursor

    Returns:
        Dictionary with 'feed' list containing posts

    Each item in 'feed' contains a 'post' dict with:
        - 'uri': Post URI (use this to fetch replies)
        - 'author': Dict with 'handle', 'displayName'
        - 'record': Dict with 'text', 'createdAt'
        - 'likeCount', 'replyCount', 'repostCount': Engagement metrics
    """
    params = {'actor': handle, 'limit': min(limit, 100)}
    if cursor:
        params['cursor'] = cursor
    return make_request('app.bsky.feed.getAuthorFeed', params)


def get_post_thread(uri, depth=50):
    """
    Get a post and its replies.

    Args:
        uri: Post URI (at:// format, from post['uri'])
        depth: How deep to fetch replies

    Returns:
        Dictionary with 'thread' containing post and replies

    IMPORTANT LIMITATION:
        The API returns at most ~200 direct replies, biased toward
        EARLIER replies (not a random sample!). For posts with 1,000+
        replies, you're seeing only ~20% of engagement.

    The response structure is nested. You'll need to write code to
    extract the reply information you need (see II.1 in the assignment).
    """
    params = {'uri': uri, 'depth': depth}
    return make_request('app.bsky.feed.getPostThread', params)


# ============================================================================
# Utility Functions
# ============================================================================

def parse_datetime(date_string):
    """
    Parse an ISO 8601 datetime string from Bluesky.

    Args:
        date_string: ISO format datetime (e.g., '2025-01-20T15:30:00.000Z')

    Returns:
        datetime object with timezone info
    """
    return datetime.fromisoformat(date_string.replace('Z', '+00:00'))


def is_within_hours(date_string, hours=24):
    """
    Check if a datetime string is within the last N hours.

    Args:
        date_string: ISO format datetime
        hours: Number of hours to check

    Returns:
        True if the datetime is within the specified window
    """
    try:
        post_time = parse_datetime(date_string)
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        return post_time >= cutoff
    except:
        return False


def load_senators(csv_file='senators_bluesky.csv'):
    """
    Load senator data from CSV.

    Args:
        csv_file: Path to senators CSV file

    Returns:
        List of dictionaries with senator info

    Each dict contains:
        - 'name': Senator's full name
        - 'handle': Bluesky handle (use this for API calls)
        - 'party': Political party ('D', 'I', etc.)
        - 'state': Two-letter state code
        - 'gender': 'F' or 'M'
    """
    import csv

    senators = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            senators.append(row)
    return senators


def save_json(data, filename):
    """
    Save data to a JSON file.

    Useful for saving collected data so you don't need to re-collect.
    Uses indent=2 for readable output.
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filename):
    """
    Load data from a JSON file.

    Use this to reload previously collected data.
    """
    with open(filename, 'r') as f:
        return json.load(f)


# ============================================================================
# Gender Inference - SCAFFOLDING
# You need to implement these functions for Part II.2 of the assignment.
# ============================================================================

def load_name_data(female_file='female_names.tsv.gz', male_file='male_names.tsv.gz'):
    """
    Load SSA baby name data for gender inference.

    Args:
        female_file: Path to female names TSV (gzipped)
        male_file: Path to male names TSV (gzipped)

    Returns:
        A data structure mapping names to their gender counts.
        The exact structure is up to you, but you'll need to be able to
        look up how many times a name was registered as female vs male.

    File format (tab-separated, with header row):
        name    count   year
        Mary    7065    1880
        Anna    2604    1880
        ...

    The files contain historical baby name registrations from the Social
    Security Administration. Names can appear in BOTH files (like "Jordan").

    Hints:
        - Use gzip.open() to read the compressed files
        - Skip the header row
        - Aggregate counts across all years for each name
        - Names should be case-insensitive (lowercase them)

    Example usage after implementation:
        name_data = load_name_data()
        # name_data should let you look up counts for any name
        # e.g., "mary" should show ~99% female registrations
        # e.g., "jordan" should show ~50/50 split
    """
    import gzip

    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    # 1. Create a data structure to store name -> [female_count, male_count]
    # 2. Read female_file and add counts (use gzip.open())
    # 3. Read male_file and add counts
    # 4. Return your data structure
    #
    # Example structure you might use:
    #   name_counts = {}  # name -> [female_count, male_count]
    # =========================================================================

    raise NotImplementedError("You need to implement load_name_data()")


def infer_gender(display_name, name_data, threshold=0.6):
    """
    Infer gender from a display name using SSA data.

    Args:
        display_name: User's display name (e.g., "John Smith", "Dr. Jane Doe")
        name_data: Data structure from load_name_data()
        threshold: Minimum proportion to classify (default 0.6 = 60%)

    Returns:
        'F' for female, 'M' for male, or 'U' for unknown

    Steps to implement:
        1. Extract a likely first name from the display name
           - Handle edge cases: empty names, titles (Dr., Sen.), etc.
           - Some display names are handles/usernames, not real names
        2. Look up the name in your name_data
        3. If >threshold of registrations are one gender, return that gender
        4. Otherwise return 'U' (unknown/ambiguous)

    The threshold determines confidence level:
        - threshold=0.6: Classify if 60%+ are one gender (more classifications)
        - threshold=0.8: Classify only if 80%+ (fewer but more confident)

    Examples (assuming threshold=0.6):
        "Mary Smith"     -> 'F' (nearly 100% female)
        "John Doe"       -> 'M' (nearly 100% male)
        "Jordan Lee"     -> 'U' (roughly 50/50)
        "xXx_user_123"   -> 'U' (not a recognizable name)
        ""               -> 'U' (empty)

    Limitations to consider (for reflection questions):
        - International names may not be in SSA data
        - Nicknames may not match ("Bob" vs "Robert")
        - Names change gender associations over time
        - Non-binary individuals may be misclassified
    """
    # =========================================================================
    # YOUR CODE HERE
    # =========================================================================
    # 1. Extract first name from display_name
    #    - Handle empty strings, titles like "Dr.", "Sen.", etc.
    #    - Consider: what if it's a username like "gamer_123"?
    #
    # 2. Look up the extracted name in name_data
    #    - Remember to handle case (lowercase?)
    #    - What if the name isn't found?
    #
    # 3. Calculate the proportion female (or male) and compare to threshold
    #    - female_ratio = female_count / (female_count + male_count)
    #
    # 4. Return 'F', 'M', or 'U'
    # =========================================================================

    raise NotImplementedError("You need to implement infer_gender()")


# ============================================================================
# Example usage - Run this file directly to test the API helpers
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Testing Bluesky API helpers")
    print("=" * 60)

    # Test basic API functions
    handle = "schumer.senate.gov"

    # Test 1: Get a profile
    print(f"\n1. Getting profile for {handle}...")
    profile = get_profile(handle)
    if profile:
        print(f"   Display name: {profile.get('displayName')}")
        print(f"   Followers: {profile.get('followersCount')}")
        print(f"   Following: {profile.get('followsCount')}")
        print(f"   Posts: {profile.get('postsCount')}")
    else:
        print("   ERROR: Could not fetch profile")

    # Test 2: Get follows (first page only for testing)
    print(f"\n2. Getting first 5 accounts {handle} follows...")
    follows = get_follows(handle, limit=5)
    if follows:
        for f in follows.get('follows', []):
            print(f"   - {f.get('displayName', f.get('handle'))}")
    else:
        print("   ERROR: Could not fetch follows")

    # Test 3: Get recent posts
    print(f"\n3. Getting 3 recent posts from {handle}...")
    feed = get_author_feed(handle, limit=3)
    if feed:
        for item in feed.get('feed', []):
            post = item.get('post', {})
            text = post.get('record', {}).get('text', '')[:60]
            replies = post.get('replyCount', 0)
            print(f"   - [{replies} replies] {text}...")
    else:
        print("   ERROR: Could not fetch feed")

    # Test 4: Get thread (show structure for students to understand)
    print(f"\n4. Getting a post thread to show structure...")
    if feed and feed.get('feed'):
        first_post = feed['feed'][0]['post']
        uri = first_post.get('uri')
        print(f"   Post URI: {uri}")
        thread = get_post_thread(uri)
        if thread and 'thread' in thread:
            replies = thread['thread'].get('replies', [])
            print(f"   Found {len(replies)} top-level replies")
            print("   Thread structure keys:", list(thread['thread'].keys()))
            if replies:
                print("   First reply keys:", list(replies[0].keys()) if replies[0] else "None")
        else:
            print("   No thread data returned (post may have no replies)")

    print("\n" + "=" * 60)
    print("API helpers working correctly!")
    print("=" * 60)
    print("\nRemember:")
    print("  - Add time.sleep(0.1) between API calls")
    print("  - Handle errors gracefully (some accounts may be deleted)")
    print("  - Save intermediate results to avoid re-collecting")
    print("\nFor gender inference, you need to implement:")
    print("  - load_name_data()")
    print("  - infer_gender()")
