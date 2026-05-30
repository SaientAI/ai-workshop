//! gguf.rs — Pure-Rust GGUF file reader.
//! Reads the GGUF container format directly from disk using memmap2.
//! Supports GGUF versions 1, 2, and 3.
//! No llama.cpp. No C. No FFI.

use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"

#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    pub fn as_u32(&self) -> Option<u32> { if let Self::Uint32(v) = self { Some(*v) } else { None } }
    pub fn as_u64(&self) -> Option<u64> { if let Self::Uint64(v) = self { Some(*v) } else { None } }
    pub fn as_str(&self) -> Option<&str> { if let Self::String(s) = self { Some(s) } else { None } }
    pub fn as_f32(&self) -> Option<f32> { if let Self::Float32(v) = self { Some(*v) } else { None } }
    pub fn as_i32(&self) -> Option<i32> { if let Self::Int32(v) = self { Some(*v) } else { None } }
    pub fn as_bool(&self) -> Option<bool> { if let Self::Bool(v) = self { Some(*v) } else { None } }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    // raw mmap kept alive
    _mmap: Mmap,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .with_context(|| format!("Cannot open {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file) }
            .context("mmap failed")?;
        Self::parse(mmap)
    }

    fn parse(mmap: Mmap) -> Result<Self> {
        let data = &mmap[..];
        let mut pos = 0usize;

        // Magic
        let magic = read_u32(data, &mut pos)?;
        if magic != GGUF_MAGIC {
            bail!("Not a GGUF file (bad magic: 0x{:08X})", magic);
        }

        // Version
        let version = read_u32(data, &mut pos)?;
        if !(1..=3).contains(&version) {
            bail!("Unsupported GGUF version: {}", version);
        }

        // Tensor count + metadata kv count
        let tensor_count = if version == 1 { read_u32(data, &mut pos)? as u64 }
                           else { read_u64(data, &mut pos)? };
        let kv_count    = if version == 1 { read_u32(data, &mut pos)? as u64 }
                           else { read_u64(data, &mut pos)? };

        // Metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..kv_count {
            let key = read_string(data, &mut pos)?;
            let val = read_value(data, &mut pos, version)?;
            metadata.insert(key, val);
        }

        // Tensor infos
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for _ in 0..tensor_count {
            let name   = read_string(data, &mut pos)?;
            let n_dims = read_u32(data, &mut pos)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(if version == 1 { read_u32(data, &mut pos)? as u64 }
                          else { read_u64(data, &mut pos)? });
            }
            let ggml_type = read_u32(data, &mut pos)?;
            let offset    = read_u64(data, &mut pos)?;
            tensors.push(TensorInfo { name, n_dims, dims, ggml_type, offset });
        }

        // Align to 32-byte boundary for data section
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as usize;
        if !pos.is_multiple_of(alignment) {
            pos += alignment - (pos % alignment);
        }
        let data_offset = pos as u64;

        Ok(Self { version, metadata, tensors, data_offset, _mmap: mmap })
    }

    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    pub fn quantization_version(&self) -> Option<u32> {
        self.metadata.get("general.quantization_version").and_then(|v| v.as_u32())
    }

    pub fn context_length(&self) -> Option<u64> {
        self.arch_u64_any(&["context_length"])
    }

    pub fn embedding_length(&self) -> Option<u64> {
        self.arch_u64_any(&["embedding_length"])
    }

    pub fn block_count(&self) -> Option<u64> {
        self.arch_u64_any(&["block_count"])
    }

    fn arch_u64_any(&self, suffixes: &[&str]) -> Option<u64> {
        let mut arches = Vec::new();
        if let Some(arch) = self.architecture() {
            arches.push(arch);
        }
        arches.extend(["llama", "mistral", "phi", "qwen2", "gemma", "falcon", "gpt-oss"]);

        for arch in arches {
            for suffix in suffixes {
                let key = format!("{}.{}", arch, suffix);
                if let Some(v) = self.metadata.get(&key) {
                    match v {
                        GgufValue::Uint32(n) => return Some(*n as u64),
                        GgufValue::Uint64(n) => return Some(*n),
                        _ => {}
                    }
                }
            }
        }
        None
    }

    pub fn file_type(&self) -> Option<u32> {
        self.metadata.get("general.file_type").and_then(|v| v.as_u32())
    }

    pub fn quant_name(&self) -> String {
        match self.file_type() {
            Some(0)  => "F32",
            Some(1)  => "F16",
            Some(2)  => "Q4_0",
            Some(3)  => "Q4_1",
            Some(6)  => "Q5_0",
            Some(7)  => "Q5_1",
            Some(8)  => "Q8_0",
            Some(9)  => "Q8_1",
            Some(10) => "Q2_K",
            Some(11) => "Q3_K_S",
            Some(12) => "Q3_K_M",
            Some(13) => "Q3_K_L",
            Some(14) => "Q4_K_S",
            Some(15) => "Q4_K_M",
            Some(16) => "Q5_K_S",
            Some(17) => "Q5_K_M",
            Some(18) => "Q6_K",
            Some(19) => "Q8_K",
            Some(20) => "IQ2_XXS",
            Some(21) => "IQ2_XS",
            Some(24) => "IQ3_XXS",
            Some(26) => "IQ4_NL",
            Some(28) => "IQ3_S",
            Some(29) => "IQ2_S",
            Some(30) => "IQ4_XS",
            Some(n)  => return format!("type_{}", n),
            None     => "unknown",
        }.to_string()
    }

    pub fn summary(&self) -> ModelSummary {
        ModelSummary {
            name: self.model_name().unwrap_or("unknown").to_string(),
            architecture: self.architecture().unwrap_or("unknown").to_string(),
            quant: self.quant_name(),
            context_length: self.context_length().unwrap_or(0),
            embedding_length: self.embedding_length().unwrap_or(0),
            block_count: self.block_count().unwrap_or(0),
            tensor_count: self.tensors.len() as u64,
            version: self.version,
        }
    }
}

#[derive(Debug, serde::Serialize, serde::Deserialize, Clone)]
pub struct ModelSummary {
    pub name: String,
    pub architecture: String,
    pub quant: String,
    pub context_length: u64,
    pub embedding_length: u64,
    pub block_count: u64,
    pub tensor_count: u64,
    pub version: u32,
}

// ── Low-level binary readers ──────────────────────────────────────────────────

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8> {
    let v = *data.get(*pos).context("unexpected EOF reading u8")?;
    *pos += 1;
    Ok(v)
}
fn read_i8(data: &[u8], pos: &mut usize) -> Result<i8> { Ok(read_u8(data, pos)? as i8) }

fn read_u16(data: &[u8], pos: &mut usize) -> Result<u16> {
    let b = data.get(*pos..*pos+2).context("unexpected EOF reading u16")?;
    *pos += 2;
    Ok(u16::from_le_bytes(b.try_into().unwrap()))
}
fn read_i16(data: &[u8], pos: &mut usize) -> Result<i16> { Ok(read_u16(data, pos)? as i16) }

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32> {
    let b = data.get(*pos..*pos+4).context("unexpected EOF reading u32")?;
    *pos += 4;
    Ok(u32::from_le_bytes(b.try_into().unwrap()))
}
fn read_i32(data: &[u8], pos: &mut usize) -> Result<i32> { Ok(read_u32(data, pos)? as i32) }

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64> {
    let b = data.get(*pos..*pos+8).context("unexpected EOF reading u64")?;
    *pos += 8;
    Ok(u64::from_le_bytes(b.try_into().unwrap()))
}
fn read_i64(data: &[u8], pos: &mut usize) -> Result<i64> { Ok(read_u64(data, pos)? as i64) }

fn read_f32(data: &[u8], pos: &mut usize) -> Result<f32> {
    let v = read_u32(data, pos)?;
    Ok(f32::from_bits(v))
}
fn read_f64(data: &[u8], pos: &mut usize) -> Result<f64> {
    let v = read_u64(data, pos)?;
    Ok(f64::from_bits(v))
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String> {
    let len = read_u64(data, pos)? as usize;
    let bytes = data.get(*pos..*pos+len)
        .with_context(|| format!("unexpected EOF reading string of len {}", len))?;
    *pos += len;
    Ok(String::from_utf8_lossy(bytes).into_owned())
}

fn read_value(data: &[u8], pos: &mut usize, version: u32) -> Result<GgufValue> {
    let type_id = read_u32(data, pos)?;
    match type_id {
        0  => Ok(GgufValue::Uint8(read_u8(data, pos)?)),
        1  => Ok(GgufValue::Int8(read_i8(data, pos)?)),
        2  => Ok(GgufValue::Uint16(read_u16(data, pos)?)),
        3  => Ok(GgufValue::Int16(read_i16(data, pos)?)),
        4  => Ok(GgufValue::Uint32(read_u32(data, pos)?)),
        5  => Ok(GgufValue::Int32(read_i32(data, pos)?)),
        6  => Ok(GgufValue::Float32(read_f32(data, pos)?)),
        7  => Ok(GgufValue::Bool(read_u8(data, pos)? != 0)),
        8  => Ok(GgufValue::String(read_string(data, pos)?)),
        9  => {
            // Array: elem_type (u32) + count (u64 v2/v3, u32 v1)
            let elem_type = read_u32(data, pos)?;
            let count = if version == 1 { read_u32(data, pos)? as u64 } else { read_u64(data, pos)? };
            let mut arr = Vec::with_capacity(count.min(65536) as usize);
            for _ in 0..count {
                // Push elem_type as if it were a value type prefix
                // We do this by temporarily prepending it
                let elem = read_value_of_type(data, pos, elem_type)?;
                arr.push(elem);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(read_u64(data, pos)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, pos)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, pos)?)),
        t  => bail!("Unknown GGUF value type: {}", t),
    }
}

fn read_value_of_type(data: &[u8], pos: &mut usize, type_id: u32) -> Result<GgufValue> {
    match type_id {
        0  => Ok(GgufValue::Uint8(read_u8(data, pos)?)),
        1  => Ok(GgufValue::Int8(read_i8(data, pos)?)),
        2  => Ok(GgufValue::Uint16(read_u16(data, pos)?)),
        3  => Ok(GgufValue::Int16(read_i16(data, pos)?)),
        4  => Ok(GgufValue::Uint32(read_u32(data, pos)?)),
        5  => Ok(GgufValue::Int32(read_i32(data, pos)?)),
        6  => Ok(GgufValue::Float32(read_f32(data, pos)?)),
        7  => Ok(GgufValue::Bool(read_u8(data, pos)? != 0)),
        8  => Ok(GgufValue::String(read_string(data, pos)?)),
        10 => Ok(GgufValue::Uint64(read_u64(data, pos)?)),
        11 => Ok(GgufValue::Int64(read_i64(data, pos)?)),
        12 => Ok(GgufValue::Float64(read_f64(data, pos)?)),
        t  => bail!("Unknown array elem type: {}", t),
    }
}
