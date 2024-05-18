#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};

use candle::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::models::blip;
use candle_transformers::models::quantized_blip;

use image::DynamicImage;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::Args;

#[derive(Debug)]
pub struct ImageAnalysisConfig {
    cpu: bool,
    quantized: bool,
    model_file: PathBuf,
    tokenizer: PathBuf,
}

pub fn image_analysis_config_from_args(args: &Args) -> Result<ImageAnalysisConfig> {
    let model_file = match &args.blip_model {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            if args.quantized {
                let api = api.model("lmz/candle-blip".to_string());
                api.get("blip-image-captioning-large-q4k.gguf")?
            } else {
                let api = api.repo(hf_hub::Repo::with_revision(
                    "Salesforce/blip-image-captioning-large".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/18".to_string(),
                ));
                api.get("model.safetensors")?
            }
        }
        Some(model) => model.into(),
    };
    let tokenizer = match &args.blip_tokenizer {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("Salesforce/blip-image-captioning-large".to_string());
            api.get("tokenizer.json")?
        }
        Some(file) => file.into(),
    };

    Ok(ImageAnalysisConfig {
        cpu: args.cpu,
        quantized: args.quantized,
        model_file,
        tokenizer,
    })
}

enum Model {
    M(blip::BlipForConditionalGeneration),
    Q(quantized_blip::BlipForConditionalGeneration),
}

impl Model {
    fn text_decoder_forward(&mut self, xs: &Tensor, img_xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::M(m) => Ok(m.text_decoder().forward(xs, img_xs)?),
            Self::Q(m) => Ok(m.text_decoder().forward(xs, img_xs)?),
        }
    }
}

const SEP_TOKEN_ID: u32 = 102;

/// Reshape the image into a tensor with shape
/// (3, 384, 384). OpenAI normalization is applied.
fn format_image(image: &DynamicImage) -> Result<Tensor> {
    let img = image.resize_to_fill(384, 384, image::imageops::FilterType::Triangle);
    let img = img.to_rgb8();
    let data = img.into_raw();
    let data = Tensor::from_vec(data, (384, 384, 3), &Device::Cpu)?.permute((2, 0, 1))?;
    let mean =
        Tensor::new(&[0.48145466f32, 0.4578275, 0.40821073], &Device::Cpu)?.reshape((3, 1, 1))?;
    let std = Tensor::new(&[0.26862954f32, 0.261_302_6, 0.275_777_1], &Device::Cpu)?
        .reshape((3, 1, 1))?;
    Ok((data.to_dtype(candle::DType::F32)? / 255.)?
        .broadcast_sub(&mean)?
        .broadcast_div(&std)?)
}

pub fn run_iteration(
    analysis_config: &ImageAnalysisConfig,
    image: &DynamicImage,
) -> anyhow::Result<String> {
    let model_file = &analysis_config.model_file;
    let tokenizer = Tokenizer::from_file(&analysis_config.tokenizer).map_err(E::msg)?;
    let mut tokenizer = TokenOutputStream::new(tokenizer);
    let mut logits_processor =
        candle_transformers::generation::LogitsProcessor::new(1337, None, None);

    let config = blip::Config::image_captioning_large();

    let device = candle_examples::device(analysis_config.cpu)?;
    let (image_embeds, device, mut model) = if analysis_config.quantized {
        let device = Device::Cpu;
        let image = format_image(image)?.to_device(&device)?;
        let vb = quantized_blip::VarBuilder::from_gguf(model_file, &device)?;
        let model = quantized_blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::Q(model))
    } else {
        let image = format_image(image)?.to_device(&device)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F32, &device)? };
        let model = blip::BlipForConditionalGeneration::new(&config, vb)?;
        let image_embeds = image.unsqueeze(0)?.apply(model.vision_model())?;
        (image_embeds, device, Model::M(model))
    };

    let mut token_ids = vec![30522u32];
    let mut desc_tokens: Vec<String> = vec![];
    for index in 0..1000 {
        let context_size = if index > 0 { 1 } else { token_ids.len() };
        let start_pos = token_ids.len().saturating_sub(context_size);
        let input_ids = Tensor::new(&token_ids[start_pos..], &device)?.unsqueeze(0)?;
        let logits = model.text_decoder_forward(&input_ids, &image_embeds)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let token = logits_processor.sample(&logits)?;
        if token == SEP_TOKEN_ID {
            break;
        }
        token_ids.push(token);
        if let Some(t) = tokenizer.next_token(token)? {
            desc_tokens.push(t);
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        desc_tokens.push(rest);
    }
    return Ok(desc_tokens.join(""));
}
