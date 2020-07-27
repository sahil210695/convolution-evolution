import numpy as np
import streamlit as st

from main import Conv2D

st.set_option('deprecation.showfileUploaderEncoding', False)

kernels = {
    "Horizontal": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
    "Vertical": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    "Edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
}


def get_readme_content():
    with open("README.md") as fr:
        return fr.read()


def action():
    st.header("Run this on image of your choice :thumbsup:")
    action_place = st.empty()

    file_upload = action_place.file_uploader("Upload Image",
                                             type=["jpg", "png"])

    if file_upload:
        selected_kernel = st.selectbox("Select Kernel", tuple(kernels.keys()))

        with st.spinner(f"Doing Convolution with {selected_kernel} Kernel"):
            convolved_image = Conv2D().process(file_upload,
                                               kernels.get(selected_kernel))
            st.image(convolved_image,
                     clamp=True,
                     caption=f"{selected_kernel} Kernel")
            st.balloons()


def main():
    st.markdown(get_readme_content())
    action()


if __name__ == "__main__":
    main()
