{
    device: "cpu",
    dataset_reader: "nmt-dataset",
    convert_to_lowercase: false,
    train_data_path: ["en_es_data/sample.es", "en_es_data/sample.en"],
    valid_data_path: ["en_es_data/sample.es", "en_es_data/sample.en"],
    max_vocab_size: 50000,
    vocab_path: "vocab",
    char_emb_size: 50,
    n_filters: 50,
    hidden_size: 256,
    max_word_length: 21,
    dropout_rate: 0.3,
    train_batch_size: 4,
    max_grad_norm: 5.0,
    lr: 0.001,
    valid_batch_size: 15,
    results_path: "results"
}