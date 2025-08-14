# oauth_handler.py

import requests
import urllib.parse
import uuid


class OAuth2Handler:
    """
    A simple OAuth2 handler that builds authorization URLs,
    exchanges authorization codes for tokens, and refreshes tokens.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        auth_base_url: str,
        token_url: str,
        scopes: list[str],
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.auth_base_url = auth_base_url
        self.token_url = token_url
        self.scopes = scopes

    def get_authorization_url(self) -> tuple[str, str]:
        """
        Construct the OAuth2 authorization URL and a state token.
        Returns a tuple of (url, state).
        """
        state = str(uuid.uuid4())
        params = {
            "client_id":     self.client_id,
            "redirect_uri":  self.redirect_uri,
            "response_type": "code",
            "scope":         " ".join(self.scopes),
            "state":         state,
        }
        url = f"{self.auth_base_url}?{urllib.parse.urlencode(params)}"
        return url, state

    def exchange_code_for_tokens(self, code: str) -> dict:
        """
        Exchange an authorization code for access and refresh tokens.
        Returns the token response as a dict.
        """
        auth = (self.client_id, self.client_secret)
        data = {
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": self.redirect_uri,
        }
        response = requests.post(self.token_url, auth=auth, data=data, timeout=10)
        response.raise_for_status()
        return response.json()

    def refresh_access_token(self, refresh_token: str) -> dict:
        """
        Use a refresh token to obtain a new access token (and possibly a new refresh token).
        Returns the token response as a dict.
        """
        auth = (self.client_id, self.client_secret)
        data = {
            "grant_type":    "refresh_token",
            "refresh_token": refresh_token,
        }
        response = requests.post(self.token_url, auth=auth, data=data, timeout=10)
        response.raise_for_status()
        return response.json()
