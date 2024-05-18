#![allow(non_snake_case)]

#[cfg(feature = "server")]
use base64::prelude::*;
use dioxus::prelude::*;
use std::collections::HashMap;
#[cfg(feature = "server")]
use std::io::Cursor;
use tracing::Level;

mod dioxus_components;
use dioxus_components::{
    SdBlipAttractionHeader, SdBlipAttractionInput, SdBlipAttractionLastResponse,
    SdBlipAttractionResponse,
};

#[cfg(feature = "server")]
mod blip;
#[cfg(feature = "server")]
use blip::image_analysis_config_from_args;

#[cfg(feature = "server")]
mod sd;

#[cfg(feature = "server")]
use sd::{image_generation_config_from_args, StableDiffusionVersion};

#[cfg(feature = "server")]
struct Args {
    cpu: bool,
    height: Option<usize>,
    width: Option<usize>,
    /// The UNet weight file, in .safetensors format.
    unet_weights: Option<String>,

    /// The CLIP weight file, in .safetensors format.
    clip_weights: Option<String>,

    /// The VAE weight file, in .safetensors format.
    vae_weights: Option<String>,

    /// The file specifying the tokenizer to used for tokenization.
    sd_tokenizer: Option<String>,

    /// The size of the sliced attention or 0 for automatic slicing (disabled by default)
    sliced_attention_size: Option<usize>,

    /// The number of steps to run the diffusion for.
    sd_n_steps: Option<usize>,

    /// The number of samples to generate.
    num_samples: u64,

    sd_version: StableDiffusionVersion,

    use_flash_attn: bool,

    use_f16: bool,

    guidance_scale: Option<f64>,

    /// The seed to use when generating random samples.
    seed: Option<u64>,
    blip_model: Option<String>,
    blip_tokenizer: Option<String>,

    /// Use the quantized version of the model.
    quantized: bool,
}

#[derive(Clone, Debug, PartialEq)]
enum AttractionStatus {
    Idle,
    Interrupted,
    Iterating,
}

#[derive(Clone, Debug, PartialEq)]
enum FixedPointKind {
    None,
    FixedPoint,
    Loop,
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SdBlipResponse {
    prompt: String,
    image_base64: String,
    description: String,
}

#[derive(Clone, Routable, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
enum Route {
    #[route("/")]
    Home {},
}

fn main() {
    // Init debug
    dioxus_logger::init(Level::INFO).expect("failed to init logger");
    // Need to listen on 0.0.0.0 for Docker
    let cfg = server_only!(
        dioxus::fullstack::Config::new().addr(std::net::SocketAddr::from(([0, 0, 0, 0], 8080)))
    );

    LaunchBuilder::fullstack().with_cfg(cfg).launch(App);
}

fn App() -> Element {
    rsx! {
        Router::<Route> {}
    }
}

#[component]
fn Home() -> Element {
    let prompt = use_signal(|| String::from(""));
    let response_map = use_signal(HashMap::<i32, SdBlipResponse>::new);
    let status = use_signal(|| AttractionStatus::Idle);
    let max_depth = 3;
    rsx! {
        div {
            class: "container mt-4",
            SdBlipAttractionHeader {max_depth: max_depth},
            SdBlipAttractionInput { prompt: prompt, response_map: response_map, status: status },
            SdBlipAttractionResponse { index: 0, response_map: response_map, status: status },
            SdBlipAttractionResponse { index: 1, response_map: response_map, status: status },
            SdBlipAttractionLastResponse { index: max_depth - 1, max_depth: max_depth, response_map: response_map, status: status },
        }
    }
}

#[server(GetServerData)]
async fn image_and_desc_for_prompt(prompt: String) -> Result<SdBlipResponse, ServerFnError> {
    println!("Server received: {}", prompt);
    let args = Args {
        cpu: false,
        height: Some(384 as usize),
        width: Some(384 as usize),
        unet_weights: None,
        clip_weights: None,
        vae_weights: None,
        sd_tokenizer: None,
        sliced_attention_size: None,
        sd_n_steps: None,
        num_samples: 1,
        sd_version: StableDiffusionVersion::Turbo,
        use_flash_attn: false,
        use_f16: false,
        guidance_scale: None,
        seed: None,
        blip_model: None,
        blip_tokenizer: None,
        quantized: false,
    };
    let image_generation_config = image_generation_config_from_args(&args);
    let image = sd::run_iteration(&prompt, &image_generation_config, &args);
    if image.is_err() {
        return Err(ServerFnError::new("Failed to generate image"));
    }
    let image = image.unwrap();
    let image_analysis_config = image_analysis_config_from_args(&args);
    if image_analysis_config.is_err() {
        return Err(ServerFnError::new(
            "Failed to create config for image analysis",
        ));
    }
    let description = blip::run_iteration(&image_analysis_config.unwrap(), &image);
    if description.is_err() {
        return Err(ServerFnError::new("Failed to analyze image"));
    }

    let mut buf: Vec<u8> = Vec::new();
    image.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)?;
    let img_base64 = BASE64_STANDARD.encode(&buf);

    let response = SdBlipResponse {
        prompt: prompt.clone(),
        image_base64: img_base64,
        description: description.unwrap(),
    };

    Ok(response)
}
