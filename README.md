---
title: Sd Blip Attraction
emoji: 💻
colorFrom: blue
colorTo: yellow
sdk: docker
pinned: false
license: apache-2.0
app_port: 8080
---

# SD-BLIP Attraction

Take a prompt, generate an image using Stable Diffusion, feed the image into BLIP to generate a description, and use that as a prompt for Stable Diffusion. Repeat this until a BLIP description is produced that has already been seen, or until we reach the maximum number of steps.

![an image of a run](support/img/run-image.png)

## Try out in the cloud

You can try it out sd-blip attraction on huggingface at:

https://huggingface.co/spaces/ciyer/sd-blip-attraction

_Be warned, though, that you will need to have some patience to see the output._


## Run on your on hardware

To run, you need the [dioxus cli](https://dioxuslabs.com/learn/0.5/getting_started) (and all its dependencies) installed. Once you have that, you can run the app with:

```bash
dx serve
```
And then connect a web browser to http://localhost:8080.

## Build

To build for distribution, run:

```
dx build --release --platform fullstack
```

The app should be in the `dist` directory, and can be run with:

```
dist/sd-blip-attraction
```

## Hardware Acceleration

The app runs much faster with hardware acceleration, but requires a slightly different command for the build.

To build using Metal on Apple hardware, use the following sequence of commands:

```
dx build --release --platform fullstack --server-feature metal
cargo build --release --features server,metal
```

And then run with:

```
target/release/sd-blip-attraction
```

Connect a web browser to http://localhost:8080, where you should see the app. You may see some complaints in the logs, but they can be ignored.

It should be possible to replace `metal` with `cuda` in the command above, but I have not tested this myself, since I do not have the right hardware.
