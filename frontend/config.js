// Local default. release.yml overwrites this file with the EC2 API URL before
// uploading the frontend to S3, so the deployed site never points at localhost.
window.API_BASE_URL = "http://localhost:8000";
