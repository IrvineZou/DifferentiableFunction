operation_to_num = {
    "ADD": 11, "MVM": 12, "SIG": 13, "MUL": 14, "SOFTMAX": 15, "CONCAT": 16, "RELU": 17,
    "InputVector": 18, "InputVectorTile": 19, "OutputVector": 21, "OutputVectorTile": 22,
    "PseudoInput": 23, "PseudoOutput": 24, "InputImagePixelStream": 25, "OutputImagePixelStream": 26,
    'InputImagePixelStreamTile': 27, 'OutputImagePixelStreamTile': 28, "OutputStreamVector": 29, 'Load': 30,
    'Store': 31, 'Send': 32,
    'Receive': 33, 'Copy': 34, 'WriteInput': 35, 'ReadOutput': 37,
    "Set": 36
}

hetro_limit = {
    i : 136 if i < 16 else 272 if i < 32 else 544  #if i < 32 else 2176
    for i in range(64)
}

hom_limit =  {i: 272 for i in range(64)}

model_save_path = ''
model_load_path = ''
