import urllib.request


# Helper function to extract HTTP status from an HTTPError.
# This is crucial as HTTPError uses .code instead of .status.
def _get_http_status(obj):
    return obj.code if hasattr(obj, 'code') else None


def get_paged_request(url, retries=5):
    while retries > 0:
        try:
            with urllib.request.urlopen(url) as response:
                status = response.getcode()
                yield response.read()
                break  # exit loop on successful request
        except urllib.error.HTTPError as f:
            status = _get_http_status(f)
            print(f'Received HTTPError: {status}')  # adjusted print statement
            retries -= 1
            if retries == 0:
                raise


# MockError class to simulate HTTP errors for testing
class MockError(Exception):
    def __init__(self):
        self.headers = {}  # Define headers as an empty dict
