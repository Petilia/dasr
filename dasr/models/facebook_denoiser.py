from denoiser import pretrained

def get_pretrained_model(name, freeze_encoder=False,  freeze_lstm=False, freeze_decoder=False):
    if name == "dns48":
        model = pretrained.dns48()
    elif name == "dns64":
        model = pretrained.dns64()
    elif name == "master64":
        model = pretrained.master64()
    elif name == "valentini_nc":    
        model = pretrained.valentini_nc()
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
    

