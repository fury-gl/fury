import requests
from requests.exceptions import HTTPError


def extract_status(response):
    """
    Extracts the status code from a response or error object.
    This ensures that we handle different cases where the status code may not be directly accessible.
    """
    if hasattr(response, 'status_code'):
        return response.status_code
    if hasattr(response, 'response') and hasattr(response.response, 'status_code'):
        return response.response.status_code
    return None


def safe_request(url):
    """
    Performs a GET request to the specified URL and safely handles HTTP errors.
    This function will catch HTTPError exceptions and return a None status code if unable to extract.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise for HTTP errors
        return response.json()
    except HTTPError as http_err:
        status = extract_status(http_err)
        print(f'HTTP error occurred: {http_err}, Status code: {status}')  # Or handle the error as needed
        return None
    except Exception as err:
        print(f'An error occurred: {err}')  # General error handling
        return None


# Mock error handling for testing purposes
class MockHTTPError(HTTPError):
    def __init__(self, message, headers=None):
        super().__init__(message)
        self.response = requests.Response()
        self.response.status_code = 400
        if headers:
            self.response.headers = headers

# Example usage of safe_request:
# result = safe_request('https://api.example.com/resource')
