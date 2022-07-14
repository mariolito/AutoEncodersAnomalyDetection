from tensorflow.keras import layers


def set_input(model, config):
    model.add(
        layers.InputLayer(
            input_shape=config.get("input_shape", None),
            batch_size=config.get("batch_size", None),
            dtype=config.get("dtype", None),
            input_tensor=config.get("input_tensor", None),
            sparse=config.get("sparse", None),
            name=config.get("name", None),
            ragged=config.get("ragged", None)
        )
    )
    return model


def _add_activation(model, activation_name, activation_config=None):
    if activation_name == "leaky_relu":
        model.add(
            layers.LeakyReLU(alpha=activation_config.get("alpha", None))
        )
    else:
        model.add(
            layers.Activation(activation_name)
        )
    return model


def create_layer_config(config_general, config_layer):
    config_general.update(config_layer)
    return {i: j for i, j in config_general.items() if i not in ['layers']}


def add_layer(model, config, prev_name):
    if prev_name.startswith("cnn") and config['name'] == "dense":
        model.add(layers.Flatten())
    if config['name'] == "cnn2d":
        model.add(
            layers.Conv2D(
                filters=config.get("filters", None),
                kernel_size=config.get("kernel_size", None),
                strides=config.get("strides", None),
                padding=config.get("padding", None),
                data_format=config.get("data_format", None),
                dilation_rate=config.get("dilation_rate", (1, 1)),
                groups=config.get("groups", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                bias_initializer=config.get("bias_initializer", None),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn2dT":
        model.add(
            layers.Conv2DTranspose(
                filters=config.get("filters", None),
                kernel_size=config.get("kernel_size", None),
                strides=config.get("strides", None),
                padding=config.get("padding", None),
                data_format=config.get("data_format", None),
                dilation_rate=config.get("dilation_rate", (1, 1)),
                groups=config.get("groups", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                bias_initializer=config.get("bias_initializer", None),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn1d":
        model.add(
            layers.Conv1D(
                filters=config.get("filters", None),
                kernel_size=config.get("kernel_size", None),
                strides=config.get("strides", None),
                padding=config.get("padding", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                bias_initializer=config.get("bias_initializer", None),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == "cnn1dT":
        model.add(
            layers.Conv1DTranspose(
                filters=config.get("filters", None),
                kernel_size=config.get("kernel_size", None),
                strides=config.get("strides", None),
                padding=config.get("padding", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                bias_initializer=config.get("bias_initializer", None),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
            )
        )
    elif config['name'] == 'lstm_layers':
        model.add(
            layers.RNN(
                layers.LSTMCell(
                    units=config.get("units", None),
                    recurrent_activation=config.get("recurrent_activation", 'sigmoid'),
                    use_bias=config.get("use_bias", None),
                    kernel_initializer=config.get("kernel_initializer", 'glorot_uniform'),
                    recurrent_initializer=config.get("recurrent_initializer", 'orthogonal'),
                    bias_initializer=config.get("bias_initializer", 'zeros'),
                    unit_forget_bias=config.get("unit_forget_bias", True),
                    kernel_regularizer=config.get("kernel_regularizer", None),
                    recurrent_regularizer=config.get("recurrent_regularizer", None),
                    bias_regularizer=config.get("bias_regularizer", None),
                    kernel_constraint=config.get("kernel_constraint", None),
                    recurrent_constraint=config.get("recurrent_constraint", None),
                    bias_constraint=config.get("bias_constraint", None),
                    dropout=config.get("dropout", 0.),
                    recurrent_dropout=config.get("recurrent_dropout", 0.),
                    implementation=config.get("implementation", 2)
                )
            )
        )
    elif config['name'] == 'lstm':
        model.add(
            layers.LSTMCell(
                units=config.get("units", None),
                activation=config.get("activation", None),
                recurrent_activation=config.get("recurrent_activation", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                recurrent_initializer=config.get("recurrent_initializer", 'orthogonal'),
                bias_initializer=config.get("bias_initializer", None),
                unit_forget_bias=config.get("unit_forget_bias", True),
                kernel_regularizer=config.get("kernel_regularizer", None),
                recurrent_regularizer=config.get("recurrent_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                recurrent_constraint=config.get("recurrent_constraint", None),
                bias_constraint=config.get("bias_constraint", None),
                dropout=config.get("dropout", 0.),
                recurrent_dropout=config.get("recurrent_dropout", 0.),
                implementation=config.get("implementation", 2)
            )
        )
    elif config['name'] == "dense":
        model.add(
            layers.Dense(
                units=config.get("units", None),
                use_bias=config.get("use_bias", None),
                kernel_initializer=config.get("kernel_initializer", None),
                bias_initializer=config.get("bias_initializer", None),
                kernel_regularizer=config.get("kernel_regularizer", None),
                bias_regularizer=config.get("bias_regularizer", None),
                activity_regularizer=config.get("activity_regularizer", None),
                kernel_constraint=config.get("kernel_constraint", None),
                bias_constraint=config.get("bias_constraint", None)
                # ,name=str(config.get("layer_name", None))
            )
        )
    model = _add_activation(
        model=model,
        activation_name=config['activation'],
        activation_config=config.get('activation_config', None)

    )
    if config['batch_normalization']:
        model.add(layers.BatchNormalization())
    if config['dropout']:
        model.add(layers.Dropout(config['dropout']))
    if config.get("reshape", None):
        model.add(layers.Reshape(config["reshape"]))
    if config.get("flatten", None):
        model.add(layers.Flatten())
    return model

