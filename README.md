# Convolution is Evolution 😎
### This is to see how a kernel will convolve over image and what will be its output after convolution

#### Kernels that we are using:
- **Horizontal** kernel will extract all *horizontal* edges
- **Vertical** kernel will extract all *vertical* edges
- **Edge** kernel will extract all *edges*

#### Demo Image
![Demo Image](assets/demo.png)

#### Kernels with their output:
1. Horizontal Kernel

```python
np.array([
    [-1, -1, -1],
    [ 0,  0,  0],
    [ 1,  1,  1],
])
```

![Horizontal Kernel](assets/h_krnl_output.jpg)

2. Vertical Kernel

```python
np.array([
    [-1,  0,  1],
    [-1,  0,  1],
    [-1,  0,  1],
])
```

![Vertical Kernel](assets/v_krnl_output.jpg)

3. Edge Kernel

```python
np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1],
])
```

![Edge Kernel](assets/e_krnl_output.jpg)

#### UI
We have created UI for this demo using **[Streamlit](https://www.streamlit.io/)**

- `pip install streamlit`
- `streamlit run app.py`