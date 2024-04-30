function nn = nnsetup(architecture)

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   
    nn.learningRate                     = 2;            
    nn.momentum                         = 0.5;          
    nn.scaling_learningRate             = 1;            
    nn.weightPenaltyL2                  = 0;            
    nn.nonSparsityPenalty               = 0;            
    nn.sparsityTarget                   = 0.05;         
    nn.inputZeroMaskedFraction          = 0;            
    nn.dropoutFraction                  = 0;            
    nn.testing                          = 0;            
    nn.output                           = 'sigm';       

    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
