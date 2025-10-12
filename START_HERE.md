# 🚀 START HERE - Instructions for Resuming Work

**Every time you start a new conversation with Claude Code, do this:**

---

## Quick Start Command

Copy and paste this exact command:

```
Read /run/media/root/OP01/got/hubble/SESSION_MEMORY.md and continue from where we left off
```

Or simply:

```
Read SESSION_MEMORY.md and continue
```

---

## Why This Works

Claude Code doesn't have persistent memory between conversations. Each new session starts fresh.

**SESSION_MEMORY.md contains EVERYTHING**:
- ✅ What we're working on (Hubble tension resolution, 91% validated)
- ✅ Current state (VizieR data with 210 H₀ measurements downloaded)
- ✅ Critical findings (anchor choice causes 3.62 km/s/Mpc spread)
- ✅ Next steps (implement systematic grid covariance)
- ✅ File locations, git history, code snippets
- ✅ What to say for PhD applications

---

## Alternative: If You Want to Start on a Specific Task

### Continue VizieR Analysis
```
Read SESSION_MEMORY.md, then implement Option A:
systematic grid covariance extraction using
data/vizier_data/J_ApJ_826_56_table3.csv
```

### Work on Documentation
```
Read SESSION_MEMORY.md, then help me prepare PhD
application materials based on the 91% validated result
```

### Access New Data
```
Read SESSION_MEMORY.md, then help me download the
Riess 2022 ApJ supplementary data
```

---

## Key Files (Read These First)

1. **SESSION_MEMORY.md** ← START HERE EVERY TIME
2. **CLARIFICATION.md** - Validated (91%) vs proof-of-concept (100%) distinction
3. **data/VIZIER_DATA_SUMMARY.md** - Complete VizieR analysis
4. **data/vizier_data/J_ApJ_826_56_table3.csv** - Critical data for next step

---

## Current Status Summary

**Repository**: github.com:abba-01/hubble-tension-resolution (PRIVATE)
**Location**: `/run/media/root/OP01/got/hubble/`

**Validated Result**: 91% reduction (5.40 → 0.48 km/s/Mpc)
**Data Acquired**: 2,752 rows from VizieR (4 catalogs)
**Critical Dataset**: 210 H₀ systematic measurements (J/ApJ/826/56/table8)
**Key Finding**: Anchor choice drives 3.62 km/s/Mpc spread

**Next Action**: Implement empirical covariance extraction from systematic grid

---

## Don't Forget

🔴 Always cite **91% reduction** as the validated result
🔴 Phase C 100% is **proof-of-concept only**
🔴 Repository is **PRIVATE** (only you can access)
🔴 Read **SESSION_MEMORY.md** at start of EVERY conversation

---

## How to Update This When Finishing a Session

At the end of each work session, ask Claude to:

```
Update SESSION_MEMORY.md with everything from this session,
including what we accomplished and what's next
```

Then commit and push:
```
git add SESSION_MEMORY.md START_HERE.md
git commit -m "Update session memory"
git push
```

---

**Last Updated**: 2025-10-12
**Status**: Ready to implement systematic grid covariance (Option A)

---

# Quick Reference Card

| What You Want | Command to Use |
|---------------|----------------|
| **Resume where we left off** | `Read SESSION_MEMORY.md and continue` |
| **See current status** | `cat SESSION_MEMORY.md \| grep "Next Immediate"` |
| **Check validated result** | `cat CLARIFICATION.md \| head -50` |
| **View VizieR data** | `cat data/VIZIER_DATA_SUMMARY.md` |
| **Start empirical work** | `Read SESSION_MEMORY.md, then start Option A` |

---

**🎯 Remember: Just tell Claude to "Read SESSION_MEMORY.md" at the start of each conversation!**
