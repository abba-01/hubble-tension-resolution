# Deploy UHA Algorithm Update

**Date:** 2025-10-24
**Task:** Deploy actual UHA merge algorithm to production
**Commit:** 972a378

## For Server Claude Code on 01u.aybllc.org

Execute these commands as root:

```bash
# 1. Navigate to application directory
cd /opt/uha-api

# 2. Pull latest changes from GitHub
git pull origin master

# 3. Verify the merge.py was updated
ls -lh app/merge.py
git log --oneline -1 app/merge.py

# 4. Restart the API service
systemctl restart uha-api

# 5. Check service status
systemctl status uha-api

# 6. Monitor logs for any errors
journalctl -u uha-api -n 50 --no-pager

# 7. Test health endpoint
curl https://api.aybllc.org/v1/health
```

## What Changed

- **File:** `app/merge.py`
- **Change:** Replaced placeholder weighted averaging with actual UHA N/U algebra algorithm
- **Formula:**
  - n_merge = (n1 + n2) / 2
  - u_merge = (u1 + u2) / 2 + |n1 - n2| / 2
- **Patent:** US 63/902,536

## Expected Result

The merge endpoint should now return:
- For Planck (67.4±0.5) + SH0ES (73.04±1.04):
  - merged_H0: 70.22
  - uncertainty: 3.60 (approximately)

## Validation

After deployment, test with:

```bash
curl -X POST https://api.aybllc.org/v1/merge \
  -H "Content-Type: application/json" \
  -H "X-API-Key: uha_live_a6328357aa6010d7a60cb6827e49da45" \
  -d '{
    "datasets": {
      "planck": {
        "H0": 67.4,
        "Omega_m": 0.315,
        "Omega_Lambda": 0.685,
        "sigma": {"H0": 0.5, "Omega_m": 0.007}
      },
      "shoes": {
        "H0": 73.04,
        "sigma_H0": 1.04
      }
    }
  }'
```

Expected output should show merged_H0 ≈ 70.22, uncertainty ≈ 3.60

---

**Status:** Ready for deployment
**Risk:** Low (algorithm tested locally, backward compatible)
