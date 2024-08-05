use std::error::Error;
use std::io::Write;

use tokenizers::Tokenizer;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;

/// This is a wrapper around a tokenizer to ensure that tokens can be returned to the user in a
/// streaming way rather than having to wait for the full decoding.
struct TokenOutputStream {
    tokenizer: Tokenizer,
    tokens: Vec<u32>,
    prev_index: usize,
    current_index: usize,
}

type E = Box<dyn Error + Send + Sync + 'static>;

impl TokenOutputStream {
    fn new(tokenizer: Tokenizer) -> Self {
        Self {
            tokenizer,
            tokens: Vec::new(),
            prev_index: 0,
            current_index: 0,
        }
    }

    fn decode(&self, tokens: &[u32]) -> Result<String, E> {
        self.tokenizer.decode(tokens, true)
    }

    // https://github.com/huggingface/text-generation-inference/blob/5ba53d44a18983a4de32d122f4cb46f4a17d9ef6/server/text_generation_server/models/model.py#L68
    fn next_token(&mut self, token: u32) -> Result<Option<String>, E> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        self.tokens.push(token);
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() && text.chars().last().unwrap().is_alphanumeric() {
            let text = text.split_at(prev_text.len());
            self.prev_index = self.current_index;
            self.current_index = self.tokens.len();
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }

    fn decode_rest(&self) -> Result<Option<String>, E> {
        let prev_text = if self.tokens.is_empty() {
            String::new()
        } else {
            let tokens = &self.tokens[self.prev_index..self.current_index];
            self.decode(tokens)?
        };
        let text = self.decode(&self.tokens[self.prev_index..])?;
        if text.len() > prev_text.len() {
            let text = text.split_at(prev_text.len());
            Ok(Some(text.1.to_string()))
        } else {
            Ok(None)
        }
    }
}

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

fn main() -> Result<(), E> {
    println!("start");
    let device = Device::new_cuda(0)?;
    let dtype = DType::BF16;
    let repeat_penalty = 1.1;
    let repeat_last_n = 128;
    let temperature = 0.8;
    let top_k = None;
    let top_p = None;
    let seed = 299792458;
    let sample_len = 10000;
    let path = "./Meta-Llama-3.1-8B-Instruct";
    let config_filename = format!("{path}/config.json");
    let tokenizer_filename = format!("{path}/tokenizer.json");

    let config: model::LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let config = config.into_config(false);
    println!("config ready");

    let mut cache = model::Cache::new(true, dtype, &config, &device)?;
    println!("cache ready");

    let tokenizer = Tokenizer::from_file(tokenizer_filename)?;
    println!("tokenizer ready");

    let safetensors_index_filename = format!("{path}/model.safetensors.index.json");
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(safetensors_index_filename)?)?;
    let mut safetensors_files = std::collections::HashSet::new();
    if let Some(serde_json::Value::Object(weight_map)) = json.get("weight_map") {
        for value in weight_map.values() {
            if let Some(file) = value.as_str() {
                safetensors_files.insert(format!("{path}/{file}"));
            }
        }
    }
    let safetensors_files = safetensors_files.into_iter().collect::<Vec<_>>();

    // Testing 4-bit quantized models:
    //let safetensors_files = vec![format!("{path}/model.safetensors")];

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&safetensors_files, dtype, &device)? };
    println!("vb ready");

    let llama = model::Llama::load(vb, &config)?;

    println!("model ready");
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });
    let prompt = DEFAULT_PROMPT;
    println!("tokenizing the prompt");
    let mut tokens = tokenizer.encode(prompt, true)?.get_ids().to_vec();
    println!("tokenized!");
    let mut tokenizer = TokenOutputStream::new(tokenizer);

    let mut logits_processor = {
        let temperature = temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (top_k, top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;

    println!("starting the inference loop");
    print!("{prompt}");
    for index in 0..sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest()? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
