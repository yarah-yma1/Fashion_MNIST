# Outputs + Images
Train: X =  (60000, 28, 28) |
Test: X =  (10000, 28, 28)

<img width="506" height="416" alt="image" src="https://github.com/user-attachments/assets/62808038-55a1-46e1-91d8-cd2a61e0f3b5" />

(60000, 28, 28, 1) |
Model: "sequential"
| Layer (type)              | Output Shape       | Param #   |
|----------------------------|--------------------|------------|
| Conv2D                    | (None, 28, 28, 64) | 1,664      |
| MaxPooling2D              | (None, 14, 14, 64) | 0          |
| Conv2D                    | (None, 14, 14, 128)| 204,928    |
| MaxPooling2D              | (None, 7, 7, 128)  | 0          |
| Conv2D                    | (None, 7, 7, 256)  | 819,456    |
| MaxPooling2D              | (None, 3, 3, 256)  | 0          |
| Flatten                   | (None, 2304)       | 0          |
| Dense                     | (None, 256)        | 590,080    |
| Dense                     | (None, 10)         | 2,570      |

Total params: 1,618,698 (6.17 MB) |
Trainable params: 1,618,698 (6.17 MB) |
Non-trainable params: 0 (0.00 B) |
