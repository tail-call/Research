local function layer(t, kind)
end

local function Linear(t)
    return layer(t, 'linear')
end

local function Relu(t)
    return layer(t, 'relu')
end

local function Network(t)
end

local function Softmax(t)
end

Network {
    inputs = 784,
    outputs = 10,
    activation = Softmax,
    layers = {
        Linear {
            outputs = 128,
            activation = Relu,
        },
        Linear {
            outputs = 64,
            activation = Relu,
        },
    }
}