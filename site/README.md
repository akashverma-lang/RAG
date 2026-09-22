---
title: Local RAG
emoji: 📄
colorFrom: gray
colorTo: orange
sdk: static
app_file: index.html
pinned: false
---

# Local RAG — the landing page

The page people are sent to. It explains what the app is and gives the one command
that installs it.

This Space is **static on purpose**. The app itself is not hosted here and cannot be:
a page on the web cannot read a folder on the visitor's computer, so a hosted copy
could only work by having them upload their documents to someone else's server
first. Installing locally is what keeps the folder where it already is.

## Deploying it

The front matter above is what Hugging Face reads: `sdk: static` serves `index.html`
as-is, with no build step and nothing to run, so the Space never sleeps in a way a
visitor would notice and costs nothing.

1. Create a Space at huggingface.co/new-space — choose **Static**.
2. Push the contents of this `site/` folder to it, or connect the GitHub repo and
   point the Space at this subdirectory.

The install commands in `index.html` already point at
[akashverma-lang/RAG](https://github.com/akashverma-lang/RAG); change them only if
you fork it somewhere else.

The public URL is then `https://<user>-<space>.hf.space`, which is the single link to
share.
