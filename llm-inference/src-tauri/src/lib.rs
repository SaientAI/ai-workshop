pub mod engine;
pub mod gguf;
pub mod imggen;
pub mod internet;
pub mod paths;
pub mod resolve;
pub mod setup;
pub mod video;  // needed so imggen (compiled into the lib target too) can resolve
                // crate::video::VideoHandle for cross-screen daemon eviction.
