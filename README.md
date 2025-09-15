---
title: Diogenic AI
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

This github links to huggingface: [CSDS553 Demo - a Hugging Face Space by essmith46er](https://huggingface.co/spaces/essmith46er/CSDS553_Demo)

The .github folder syncs github with huggingface and sends notifications to discord

The Cm folder contains the structure for the philosophers and the user interface 

The Src folder conation files handling the user interface and logic for talking to the model 

The tests folder contains all test files.

Push!!! (#4)
