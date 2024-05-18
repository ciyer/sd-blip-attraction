use anyhow::{anyhow, Error as E, Result};

use candle::{DType, Device, IndexOp, Module, Tensor, D};
use candle_transformers::models::stable_diffusion;
use image::imageops::FilterType;
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::Args;

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
pub(crate) enum StableDiffusionVersion {
    V1_5,
    V2_1,
    Xl,
    Turbo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelFile {
    Tokenizer,
    Tokenizer2,
    Clip,
    Clip2,
    Unet,
    Vae,
}

#[derive(Debug)]
pub struct ImageGenerationConfig {
    dtype: DType,
    guidance_scale: f64,
    n_steps: usize,
    sd_config: stable_diffusion::StableDiffusionConfig,
    device: Device,
    use_guide_scale: bool,
    which: Vec<bool>,
}

pub fn image_generation_config_from_args(args: &Args) -> ImageGenerationConfig {
    let Args {
        cpu,
        height,
        width,
        sd_n_steps,
        sliced_attention_size,
        sd_version,
        use_f16,
        guidance_scale,
        seed,
        ..
    } = args;

    let dtype = if *use_f16 { DType::F16 } else { DType::F32 };

    let guidance_scale = sd_version.guidance_scale(*guidance_scale);
    let n_steps = sd_version.n_steps(*sd_n_steps);
    let sd_config = sd_version.config(*sliced_attention_size, *height, *width);
    let device = candle_examples::device(*cpu).unwrap();
    if let Some(seed) = seed {
        device.set_seed(*seed).unwrap();
    }
    let use_guide_scale = guidance_scale > 1.0;

    let which = match sd_version {
        StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => vec![true, false],
        _ => vec![true],
    };

    ImageGenerationConfig {
        dtype,
        guidance_scale,
        n_steps,
        sd_config,
        device,
        use_guide_scale,
        which,
    }
}

impl StableDiffusionVersion {
    pub(crate) fn config(
        &self,
        sliced_attention_size: Option<usize>,
        height: Option<usize>,
        width: Option<usize>,
    ) -> stable_diffusion::StableDiffusionConfig {
        match self {
            Self::V1_5 => {
                stable_diffusion::StableDiffusionConfig::v1_5(sliced_attention_size, height, width)
            }
            Self::V2_1 => {
                stable_diffusion::StableDiffusionConfig::v2_1(sliced_attention_size, height, width)
            }
            Self::Xl => {
                stable_diffusion::StableDiffusionConfig::sdxl(sliced_attention_size, height, width)
            }
            Self::Turbo => stable_diffusion::StableDiffusionConfig::sdxl_turbo(
                sliced_attention_size,
                height,
                width,
            ),
        }
    }

    pub(crate) fn guidance_scale(&self, guidance_scale: Option<f64>) -> f64 {
        match guidance_scale {
            Some(guidance_scale) => guidance_scale,
            None => match self {
                Self::V1_5 | Self::V2_1 | Self::Xl => 7.5,
                Self::Turbo => 0.,
            },
        }
    }

    pub(crate) fn n_steps(&self, n_steps: Option<usize>) -> usize {
        match n_steps {
            Some(n_steps) => n_steps,
            None => match self {
                Self::V1_5 | Self::V2_1 | Self::Xl => 30,
                Self::Turbo => 1,
            },
        }
    }

    fn repo(&self) -> &'static str {
        match self {
            Self::Xl => "stabilityai/stable-diffusion-xl-base-1.0",
            Self::V2_1 => "stabilityai/stable-diffusion-2-1",
            Self::V1_5 => "runwayml/stable-diffusion-v1-5",
            Self::Turbo => "stabilityai/sdxl-turbo",
        }
    }

    fn unet_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "unet/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "unet/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn vae_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "vae/diffusion_pytorch_model.fp16.safetensors"
                } else {
                    "vae/diffusion_pytorch_model.safetensors"
                }
            }
        }
    }

    fn clip_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder/model.fp16.safetensors"
                } else {
                    "text_encoder/model.safetensors"
                }
            }
        }
    }

    fn clip2_file(&self, use_f16: bool) -> &'static str {
        match self {
            Self::V1_5 | Self::V2_1 | Self::Xl | Self::Turbo => {
                if use_f16 {
                    "text_encoder_2/model.fp16.safetensors"
                } else {
                    "text_encoder_2/model.safetensors"
                }
            }
        }
    }
}

impl ModelFile {
    fn get(
        &self,
        filename: &Option<String>,
        version: StableDiffusionVersion,
        use_f16: bool,
    ) -> Result<std::path::PathBuf> {
        use hf_hub::api::sync::Api;
        match filename {
            Some(filename) => Ok(std::path::PathBuf::from(filename)),
            None => {
                let (repo, path) = match self {
                    Self::Tokenizer => {
                        let tokenizer_repo = match version {
                            StableDiffusionVersion::V1_5 | StableDiffusionVersion::V2_1 => {
                                "openai/clip-vit-base-patch32"
                            }
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo => {
                                // This seems similar to the patch32 version except some very small
                                // difference in the split regex.
                                "openai/clip-vit-large-patch14"
                            }
                        };
                        (tokenizer_repo, "tokenizer.json")
                    }
                    Self::Tokenizer2 => {
                        ("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", "tokenizer.json")
                    }
                    Self::Clip => (version.repo(), version.clip_file(use_f16)),
                    Self::Clip2 => (version.repo(), version.clip2_file(use_f16)),
                    Self::Unet => (version.repo(), version.unet_file(use_f16)),
                    Self::Vae => {
                        // Override for SDXL when using f16 weights.
                        // See https://github.com/huggingface/candle/issues/1060
                        if matches!(
                            version,
                            StableDiffusionVersion::Xl | StableDiffusionVersion::Turbo,
                        ) && use_f16
                        {
                            (
                                "madebyollin/sdxl-vae-fp16-fix",
                                "diffusion_pytorch_model.safetensors",
                            )
                        } else {
                            (version.repo(), version.vae_file(use_f16))
                        }
                    }
                };
                let filename = Api::new()?.model(repo.to_string()).get(path)?;
                Ok(filename)
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn text_embeddings(
    prompt: &str,
    tokenizer: &Option<String>,
    clip_weights: &Option<String>,
    sd_version: StableDiffusionVersion,
    sd_config: &stable_diffusion::StableDiffusionConfig,
    use_f16: bool,
    device: &Device,
    dtype: DType,
    use_guide_scale: bool,
    first: bool,
) -> Result<Tensor> {
    let tokenizer_file = if first {
        ModelFile::Tokenizer
    } else {
        ModelFile::Tokenizer2
    };
    let tokenizer = tokenizer_file.get(tokenizer, sd_version, use_f16)?;
    // let tokenizer_file_name = tokenizer.file_name().unwrap();
    let tokenizer = Tokenizer::from_file(tokenizer).map_err(E::msg)?;
    let pad_id = match &sd_config.clip.pad_with {
        Some(padding) => *tokenizer.get_vocab(true).get(padding.as_str()).unwrap(),
        None => *tokenizer.get_vocab(true).get("<|endoftext|>").unwrap(),
    };
    if first {
        println!("Running with prompt \"{prompt}\".");
    }
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    while tokens.len() < sd_config.clip.max_position_embeddings {
        tokens.push(pad_id)
    }
    let tokens = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;

    if first {
        println!("Building the Clip transformer.");
    }
    let clip_weights_file = if first {
        ModelFile::Clip
    } else {
        ModelFile::Clip2
    };
    let clip_weights = clip_weights_file.get(clip_weights, sd_version, false)?;
    let clip_config = if first {
        &sd_config.clip
    } else {
        sd_config.clip2.as_ref().unwrap()
    };
    let text_model =
        stable_diffusion::build_clip_transformer(clip_config, clip_weights, device, DType::F32)?;
    let text_embeddings = text_model.forward(&tokens)?;

    let text_embeddings = if use_guide_scale {
        let mut uncond_tokens = tokenizer
            .encode("".to_string(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        while uncond_tokens.len() < sd_config.clip.max_position_embeddings {
            uncond_tokens.push(pad_id)
        }

        let uncond_tokens = Tensor::new(uncond_tokens.as_slice(), device)?.unsqueeze(0)?;
        let uncond_embeddings = text_model.forward(&uncond_tokens)?;

        Tensor::cat(&[uncond_embeddings, text_embeddings], 0)?.to_dtype(dtype)?
    } else {
        text_embeddings.to_dtype(dtype)?
    };
    Ok(text_embeddings)
}

fn tensor_to_image(img: &Tensor, nwidth: u32, nheight: u32) -> Result<DynamicImage> {
    let (channel, height, width) = img.dims3()?;
    if channel != 3 {
        return Err(anyhow!(
            "tensor_to_image expects an input of shape (3, height, width)"
        ));
    }
    let img = img.permute((1, 2, 0))?.flatten_all()?;
    let pixels = img.to_vec1::<u8>()?;
    let image: image::ImageBuffer<image::Rgb<u8>, Vec<u8>> =
        match image::ImageBuffer::from_raw(width as u32, height as u32, pixels) {
            Some(image) => image,
            None => return Err(anyhow!("Could not convert tesor to image")),
        };
    Ok(DynamicImage::from(image).resize(nwidth, nheight, FilterType::Nearest))
}

pub fn run_iteration(
    prompt: &str,
    config: &ImageGenerationConfig,
    args: &Args,
) -> Result<DynamicImage> {
    let Args {
        sd_tokenizer,
        clip_weights,
        vae_weights,
        unet_weights,
        ..
    } = args;

    let Args {
        num_samples,
        sd_version,
        use_f16,
        use_flash_attn,
        ..
    } = *args;

    let dtype = config.dtype;
    let guidance_scale = config.guidance_scale;
    let n_steps = config.n_steps;
    let sd_config = &config.sd_config;

    let scheduler = sd_config.build_scheduler(n_steps)?;
    let device = &config.device;
    let use_guide_scale = config.use_guide_scale;

    let which = &config.which;
    let text_embeddings = which
        .iter()
        .map(|first| {
            text_embeddings(
                &prompt,
                sd_tokenizer,
                clip_weights,
                sd_version,
                &sd_config,
                use_f16,
                &device,
                dtype,
                use_guide_scale,
                *first,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let text_embeddings = Tensor::cat(&text_embeddings, D::Minus1)?;
    println!("Building the autoencoder.");
    let vae_weights = ModelFile::Vae.get(vae_weights, sd_version, use_f16)?;
    let vae = sd_config.build_vae(vae_weights, &device, dtype)?;
    println!("Building the unet.");
    let unet_weights = ModelFile::Unet.get(unet_weights, sd_version, use_f16)?;
    let unet = sd_config.build_unet(unet_weights, &device, 4, use_flash_attn, dtype)?;

    let t_start = 0;
    let bsize = 1;

    let vae_scale = match sd_version {
        StableDiffusionVersion::V1_5
        | StableDiffusionVersion::V2_1
        | StableDiffusionVersion::Xl => 0.18215,
        StableDiffusionVersion::Turbo => 0.13025,
    };

    let mut result_image: Option<Tensor> = None;
    for _idx in 0..num_samples {
        let timesteps = scheduler.timesteps();
        let latents = Tensor::randn(
            0f32,
            1f32,
            (bsize, 4, sd_config.height / 8, sd_config.width / 8),
            &device,
        )?;
        // scale the initial noise by the standard deviation required by the scheduler
        let latents = (latents * scheduler.init_noise_sigma())?;
        let mut latents = latents.to_dtype(dtype)?;

        println!("starting sampling");
        let start_time = std::time::Instant::now();
        for (timestep_index, &timestep) in timesteps.iter().enumerate() {
            if timestep_index < t_start {
                continue;
            }
            let latent_model_input = if use_guide_scale {
                Tensor::cat(&[&latents, &latents], 0)?
            } else {
                latents.clone()
            };

            let latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)?;
            let noise_pred =
                unet.forward(&latent_model_input, timestep as f64, &text_embeddings)?;

            let noise_pred = if use_guide_scale {
                let noise_pred = noise_pred.chunk(2, 0)?;
                let (noise_pred_uncond, noise_pred_text) = (&noise_pred[0], &noise_pred[1]);

                (noise_pred_uncond + ((noise_pred_text - noise_pred_uncond)? * guidance_scale)?)?
            } else {
                noise_pred
            };

            latents = scheduler.step(&noise_pred, timestep, &latents)?;
        }

        let dt = start_time.elapsed().as_secs_f32();
        println!("sampling took {:.2}s", dt);
        println!("Generating the image.");
        let image = vae.decode(&(&latents / vae_scale)?)?;
        let image = ((image / 2.)? + 0.5)?.to_device(&Device::Cpu)?;
        let image = (image.clamp(0f32, 1.)? * 255.)?.to_dtype(DType::U8)?.i(0)?;
        result_image = Some(image);
    }
    return tensor_to_image(&result_image.unwrap(), 128, 128);
}
