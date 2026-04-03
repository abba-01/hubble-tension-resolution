#!/bin/bash
# Quick deployment script for Server Claude Code on 01u.aybllc.org
# Run as root: bash /got/DEPLOY_NOW.sh

set -e

echo "=== Deploying UHA API Update ==="
echo ""
echo "Changes:"
echo "  - Implemented production algorithm"
echo "  - Obfuscated code for IP protection"
echo "  - Updated API descriptions"
echo ""

# Navigate to application directory
cd /opt/uha-api

# Pull latest changes
echo "1. Pulling latest changes from GitHub..."
git pull origin master

# Verify update
echo ""
echo "2. Latest commit:"
git log --oneline -1

# Restart service
echo ""
echo "3. Restarting uha-api service..."
systemctl restart uha-api

# Wait for service to start
sleep 2

# Check status
echo ""
echo "4. Service status:"
systemctl status uha-api --no-pager -l | head -15

# Test health endpoint
echo ""
echo "5. Testing health endpoint..."
curl -s https://api.aybllc.org/v1/health | python3 -m json.tool || echo "Health check failed"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Next: Test the merge endpoint to verify algorithm is working"
