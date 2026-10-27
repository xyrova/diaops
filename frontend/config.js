// Local default. release.yml overwrites this file with the EC2 API URL before
// uploading the frontend to S3, so the deployed site never points at localhost.
window.API_BASE_URL = "http://ec2-13-234-2-174.ap-south-1.compute.amazonaws.com:8000";
