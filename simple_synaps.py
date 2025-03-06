import numpy as NP

def sigmoid(x):
    return 1 / (1 + NP.exp(-x))

def sigmoid_moshtagh(x):
    return x * (1 - x)

builder_inputs = NP.array([[0, 0, 1],
                           [1, 1, 1],
                           [1, 0, 1],
                           [0, 1, 1]]) # این داده‌های آموزشی است
builder_outputs = NP.array([[0, 1, 1, 0]]).T # خروجی داده‌های آموزشی

synaps_weight = NP.ones((3, 1))#وزن سیناپس ها رو تعریف میکنیم
print(synaps_weight)

for i in range(100000): # بخش آموزش
    input_layers = builder_inputs#ورودی آموزشی به یک متغیر داده میشود
    output = sigmoid(NP.dot(input_layers, synaps_weight))#ضرب داخلی دو ماتریس وزن سیناپس و ورودی را به تابع سیگموید ارسال میکند
    error = builder_outputs - output#ارور های نورون رو بررسی میکند
    adjustment = error * sigmoid_moshtagh(output)#ارور هارو در مشتق تابع سیگموید که خروجی به عنوان ورودی بهش ارسال شده ضرب میکند
    synaps_weight += NP.dot(input_layers.T, adjustment)#تصحیح خطاهای نورون

print("weight of synaps after train:\n", synaps_weight)
print("output:\n", output)
