# Intro-to-Robot-Learning-
Final Project for IRL w/ Siddarth Ashok

## Website (GitHub Pages)
This repository now includes a project website:
- `index.html`
- `styles.css`
- `videos/` (place random-agent videos here)

### Publish on GitHub Pages
1. Push this repository to `main`.
2. Open repository settings on GitHub.
3. Go to `Pages`.
4. Set:
   - Source: `Deploy from a branch`
   - Branch: `main`
   - Folder: `/ (root)`
5. Save and wait for deployment.

Your site URL will be:
`https://leos-bit.github.io/Intro-to-Robot-Learning-/`

### Add random-agent videos
Put MP4 files in `videos/` using these names:
- `videos/random_run_01.mp4`
- `videos/random_run_02.mp4`
- `videos/random_run_03.mp4`

## Mac startup instructions
- `gz sim -s quad_world.sdf`
- `gz sim -g`
- `conda activate gz-ws`
- `python3 random_policy.py`
