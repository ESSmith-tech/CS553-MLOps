---
title: CSDS553 Demo
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.42.0
app_file: src/app.py
pinned: false
hf_oauth: true
hf_oauth_scopes:
- inference-api
---

When a developer pushes something to GitHub, it is set up to run tests and sync with Huggingface. If a developer attempts to push a change to GitHub that fails the test or errors while compiling, it will abort the push to HuggingFace, so any poor code that would impact the product doesn’t get to the clients. A message will be sent to our team's Discord server telling us how many of the tests failed and which tests failed.

If all of the tests pass, then GitHub will sync. A message will be sent to the teams that all tests passed and the product was synced with Hugging Face.

Push!!! (#4)
