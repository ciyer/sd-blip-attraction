ARG VARIANT="1.78-slim-bullseye"
FROM rust:${VARIANT}
ENV DEBIAN_FRONTEND noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive
RUN apt-get -qq install build-essential pkg-config libssl-dev
RUN cargo install dioxus-cli

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Set home to the user's home directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user Cargo.toml Dioxus.toml $HOME/app/
COPY --chown=user ./src $HOME/app/src/
COPY --chown=user ./assets $HOME/app/assets/
RUN dx build --release --platform fullstack
RUN chown -R user:user $HOME/app/

# Switch to the "user" user
USER user
CMD ["/home/user/app/dist/sd-blip-attraction"]
