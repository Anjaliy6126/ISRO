import os
import ee


def initialize_earth_engine(project=None, service_account_file=None, force_service_account=False, authenticate_if_needed=True):
    """Initialize Google Earth Engine with an optional GCP project or service account."""
    if service_account_file:
        if not os.path.exists(service_account_file):
            if force_service_account:
                raise FileNotFoundError(f"Service account file not found: {service_account_file}")
            if authenticate_if_needed:
                print(f"Service account file not found: {service_account_file}. Falling back to interactive Earth Engine auth.")
                service_account_file = None
            else:
                raise FileNotFoundError(f"Service account file not found: {service_account_file}")

    if service_account_file:
        try:
            from google.oauth2 import service_account
        except ImportError as exc:
            raise ImportError(
                "google-auth is required for service account authentication. "
                "Install it via pip: pip install google-auth"
            ) from exc

        credentials = service_account.Credentials.from_service_account_file(service_account_file)
        ee.Initialize(credentials=credentials, project=project)
        return

    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        return
    except ee.EEException as exc:
        message = str(exc).lower()
        if "no project found" in message or "no project specified" in message:
            if not authenticate_if_needed:
                raise
            ee.Authenticate()
            if project:
                ee.Initialize(project=project)
            else:
                ee.Initialize()
        else:
            raise
