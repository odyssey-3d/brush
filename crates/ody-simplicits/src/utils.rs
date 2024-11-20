use burn::prelude::*;

pub fn calculate_determinant<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let tensor = tensor.reshape([-1, 1, 9]).squeeze::<2>(1);
    let batch = tensor.shape().dims[0];
    let a = (0..9)
        .map(|i| tensor.clone().slice([0..batch, i..i + 1]))
        .collect::<Vec<_>>();

    let det = a[0].clone() * (a[4].clone() * a[8].clone() - a[5].clone() * a[7].clone())   // col0
      - a[1].clone() * (a[3].clone() * a[8].clone() - a[5].clone() * a[6].clone())
        + a[2].clone() * (a[3].clone() * a[7].clone() - a[4].clone() * a[6].clone());

    let det = det.unsqueeze_dim::<3>(1);
    det
}

pub fn calculate_trace<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = &tensor.device();
    let eye = Tensor::<B, 2>::eye(3, device)
        .reshape::<2, _>([1, 9])
        .unsqueeze();
    let tensor = tensor.reshape::<3, _>([-1, 1, 9]);
    let trace = (tensor * eye).sum_dim(2);
    trace
}

fn generate_2d_range(from: usize, to: usize) -> impl Iterator<Item = (usize, usize)> {
    (from..to).flat_map(move |a| (from..to).map(move |b| (a, b)))
}

pub fn calculate_inverse<B: Backend>(tensor: Tensor<B, 3>) -> Tensor<B, 3> {
    let det = calculate_determinant(tensor.clone());

    let tensor = tensor.reshape([-1, 1, 9]).squeeze::<2>(1);
    let batch = tensor.shape().dims[0];

    let a = (0..9)
        .map(|i| tensor.clone().slice([0..batch, i..i + 1]))
        .collect::<Vec<_>>();

    let adj = generate_2d_range(0, 3)
        .map(|(i, j)| {
            let mut x = vec![0, 1, 2];
            let mut y = vec![0, 1, 2];
            x.remove(i);
            y.remove(j);

            let indices = x
                .iter()
                .flat_map(|x| y.iter().map(move |y| x * 3 + y))
                .collect::<Vec<_>>();
            a[indices[0]].clone() * a[indices[3]].clone()
                - a[indices[1]].clone() * a[indices[2]].clone()
        })
        .collect::<Vec<_>>();

    let adj = Tensor::cat(adj, 1);
    let sign = Tensor::<B, 2>::from_floats(
        [[1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]],
        &adj.device(),
    );

    let adj = adj * sign;
    let adj = adj.reshape([batch, 3, 3]).swap_dims(1, 2);
    let adj = adj / det;

    adj
}
