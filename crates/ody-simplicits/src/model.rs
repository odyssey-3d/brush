use burn::{
    module::Module,
    nn::{LeakyRelu, Linear, LinearConfig},
    prelude::*,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};

#[derive(Module, Debug)]
pub struct LinearBlock<B: Backend> {
    linear1: Linear<B>,
    activation: LeakyRelu,
}

impl<B: Backend> LinearBlock<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.activation.forward(self.linear1.forward(input))
    }
}

#[derive(Config, Debug)]
pub struct LinearBlockConfig {
    in_features: usize,
    out_features: usize,

    #[config(default = "0.01")]
    leaky_slope: f64,
}

impl LinearBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearBlock<B> {
        LinearBlock {
            linear1: LinearConfig::new(self.in_features, self.out_features).init(device),
            activation: LeakyRelu {
                negative_slope: self.leaky_slope,
            },
        }
    }
}

#[derive(Module, Debug)]
pub struct SimplicitsModel<B: Backend> {
    linear1: LinearBlock<B>,
    fully_connected: Vec<LinearBlock<B>>,
    output: LinearBlock<B>,
}

impl<B: Backend> SimplicitsModel<B> {
    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = self.linear1.forward(input);
        for fc in &self.fully_connected {
            x = fc.forward(x);
        }
        self.output.forward(x)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {
    spatial_dimensions: usize,
    layer_width: usize,
    num_handles: usize,
    num_layers: usize,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> SimplicitsModel<B> {
        SimplicitsModel {
            linear1: LinearBlockConfig::new(self.spatial_dimensions, self.layer_width).init(device),
            fully_connected: (0..self.num_layers)
                .map(|_| LinearBlockConfig::new(self.layer_width, self.layer_width).init(device))
                .collect(),
            output: LinearBlockConfig::new(self.layer_width, self.num_handles).init(device),
        }
    }
}

pub fn create_model<B: Backend>(num_handles: usize, device: &B::Device) -> SimplicitsModel<B> {
    ModelConfig {
        spatial_dimensions: 3,
        layer_width: 64,
        num_handles,
        num_layers: 6,
    }
    .init(device)
}

pub fn save_simplicits_model<B: Backend>(model: &SimplicitsModel<B>, model_path: &str) {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    model
        .clone()
        .save_file(model_path, &recorder)
        .expect("Should be able to save the model weights to the provided file");
    println!("Saved simplicits model to {}", model_path);
}

pub fn load_simplicits_model<B: Backend>(
    model_path: &str,
    device: &B::Device,
) -> SimplicitsModel<B> {
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let model = create_model(10, device)
        .clone()
        .load_file(model_path, &recorder, device)
        .expect("Should be able to load the model weights from the provided file");
    println!("Loaded simplicits model from {}", model_path);
    println!("{}", model);
    model
}
