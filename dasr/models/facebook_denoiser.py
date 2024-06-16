from denoiser import pretrained

def get_pretrained_model(name,
                         pretrained_weights=True, 
                         freeze_encoder=False,
                         freeze_lstm=False,
                         freeze_decoder=False):
    if name == "dns48":
        model = pretrained.dns48(pretrained=pretrained_weights)
    elif name == "dns64":
        model = pretrained.dns64(pretrained=pretrained_weights)
    elif name == "master64":
        model = pretrained.master64(pretrained=pretrained_weights)
    elif name == "valentini_nc":    
        model = pretrained.valentini_nc(pretrained=pretrained_weights)
    else:
        raise ValueError("Unknown model name")
    
    if freeze_encoder:
        print("Freezing encoder")
        for param in model.encoder.parameters():
            param.requires_grad = False
            
    if freeze_lstm:
        print("Freezing lstm")
        for param in model.lstm.parameters():
            param.requires_grad = False
            
    if freeze_decoder:
        print("Freezing decoder")
        for param in model.decoder.parameters():
            param.requires_grad = False
    
    return model
    

