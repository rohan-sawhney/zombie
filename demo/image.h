// This file contains helper functions for reading and writing images in PNG and PFM format.

#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <functional>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <array>

template <size_t DIM>
using Array = Eigen::Array<float, DIM, 1>;
using Array3 = Array<3>;
using Vector2 = Eigen::Matrix<float, 2, 1>;

template <size_t CHANNELS>
struct Image {
    static_assert(CHANNELS == 1 || CHANNELS == 3, "Image can either have 1 or 3 channels");

    // constructors
    Image(): h(0), w(0) {};
    Image(const std::string& filename) { read(filename); };
    Image(int h, int w): buffer(h*w, Array<CHANNELS>(0)), h(h), w(w) {};

    // getters
    Array<CHANNELS> get(Vector2 uv, bool flipY=true) const;
    Array<CHANNELS>& get(size_t i, size_t j);
    float& get(size_t i, size_t j, size_t c);

    // set image values from RGB values
    void setFromRGB(size_t i, size_t j, const Array3& rgb);

    // read image from file
    void read(const std::string& filename);

    // write image to file
    void write(const std::string& filename);

    // read image from PFM file
    void readPFM(const std::string& filename);

    // read image from PNG file
    void readPNG(const std::string& filename);

    // write image to PFM file
    void writePFM(const std::string& filename);

    // write image to PNG file
    void writePNG(const std::string& filename);

    // members
    int h, w;

protected:
    // check if file has extension
    bool hasExtension(const std::string& filename, const std::string& targetExtension);

    // check if machine is little endian
    bool machineIsLittleEndian();

    // reverse byte order of RGB array
    Array3 reverseRGBByteOrder(Array3 rgb);

    // reverse byte order of float
    float reverseFloatByteOrder(float val);

    // member
    std::vector<Array<CHANNELS>> buffer;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <size_t CHANNELS>
Array<CHANNELS> Image<CHANNELS>::get(Vector2 uv, bool flipY) const
{
    int i = std::clamp(int(uv.y()*h), 0, h - 1);
    int j = std::clamp(int(uv.x()*w), 0, w - 1);
    i = flipY ? h - i - 1 : i;

    return buffer[i*w + j];
}

template <size_t CHANNELS>
Array<CHANNELS>& Image<CHANNELS>::get(size_t i, size_t j)
{
    if (i*w + j < buffer.size()) return buffer[i*w + j];
    std::cerr << "index out of range: (" << i << "," << j << ")" << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t CHANNELS>
float& Image<CHANNELS>::get(size_t i, size_t j, size_t c)
{
    if (i*w + j < buffer.size() && c < CHANNELS) return buffer[i*w + j][c];
    std::cerr << "index out of range: (" << i << "," << j << "," << c << ")" << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t CHANNELS>
void Image<CHANNELS>::setFromRGB(size_t i, size_t j, const Array3& rgb)
{
    if (CHANNELS == 1) {
        // grayscale
        get(i, j, 0) = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2];

    } else {
        // rgb
        get(i, j, 0) = rgb[0];
        get(i, j, 1) = rgb[1];
        get(i, j, 2) = rgb[2];
    }
}

template <size_t CHANNELS>
void Image<CHANNELS>::read(const std::string& filename)
{
    if (hasExtension(filename, "pfm"))  {
        return readPFM(filename);
    }

    if (hasExtension(filename, "png")) {
        return readPNG(filename);
    }

    std::cerr << filename << " not supported. Use PNG or PFM file format." << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t CHANNELS>
void Image<CHANNELS>::write(const std::string& filename)
{
    if (hasExtension(filename, "pfm")) {
        return writePFM(filename);
    }

    return writePNG(filename);
}

template <size_t CHANNELS>
void Image<CHANNELS>::readPFM(const std::string& filename)
{
    std::ifstream file;
    file.open(filename.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    char type;
    file >> type;
    if (type != 'P') {
        std::cerr << "Invalid PFM file detected while reading " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    file >> type;
    int nChannels = (type == 'F') ? 3 : 1;

    file >> w >> h;
    buffer.resize(w*h, Array<CHANNELS>(0.0f));

    float scale;
    file >> scale;
    bool fileIsLittleEndian = (scale < 0);
    bool flipByteOrder = fileIsLittleEndian != machineIsLittleEndian();

    file.ignore(1);
    std::vector<float> tmpBuffer(w*h*nChannels);
    file.read(reinterpret_cast<char*>(tmpBuffer.data()), tmpBuffer.size()*sizeof(float));

    Array3 rgb;
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            if (nChannels == 3) {
                rgb[0] = tmpBuffer[3*(i*w + j) + 0];
                rgb[1] = tmpBuffer[3*(i*w + j) + 1];
                rgb[2] = tmpBuffer[3*(i*w + j) + 2];

            } else {
                rgb = Array3(tmpBuffer[i*w + j]);
            }

            rgb = flipByteOrder ? reverseRGBByteOrder(rgb) : rgb;
            setFromRGB(i, j, rgb);
        }
    }
}

template <size_t CHANNELS>
void Image<CHANNELS>::readPNG(const std::string& filename)
{
    int channels;
    unsigned char *tmpBuffer = stbi_load(filename.c_str(), &w, &h, &channels, CHANNELS);
    if (tmpBuffer == nullptr && stbi_failure_reason()) {
        stbi_image_free(tmpBuffer);
        std::cerr << "Error opening file: " << filename << std::endl;
        std::cerr << stbi_failure_reason() << std::endl;
        exit(EXIT_FAILURE);
    }

    buffer.resize(w*h, Array<CHANNELS>(0.0f));

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < CHANNELS; c++) {
                get(i, j, c) = int(tmpBuffer[CHANNELS*(i*w + j) + c])/255.0f;
            }
        }
    }

    stbi_image_free(tmpBuffer);
}

template <size_t CHANNELS>
void Image<CHANNELS>::writePFM(const std::string& filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    if (CHANNELS == 3) {
        file << "PF" << std::endl;

    } else {
        file << "Pf" << std::endl;
    }

    file << w << " " << h << std::endl;
    file << (machineIsLittleEndian() ? "-1" : "1") << std::endl;

    std::vector<float> tmpBuffer(w*h*CHANNELS);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < CHANNELS; c++) {
                int flipped_i = h - i - 1;
                tmpBuffer[CHANNELS*(flipped_i*w + j) + c] = get(i, j, c);
            }
        }
    }

    file.write(reinterpret_cast<const char*>(tmpBuffer.data()), tmpBuffer.size()*sizeof(float));
}

template <size_t CHANNELS>
void Image<CHANNELS>::writePNG(const std::string& filename)
{
    std::vector<unsigned char> tmpBuffer(h*w*CHANNELS);
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            for (int c = 0; c < CHANNELS; c++) {
                int val = std::clamp(int(get(i, j, c)*255.0f), 0, 255);
                tmpBuffer[CHANNELS*(i*w + j) + c] = static_cast<unsigned char>(val);
            }
        }
    }

    if (!stbi_write_png(filename.c_str(), w, h, CHANNELS, tmpBuffer.data(), w*3)) {
        std::cerr << "Failed to save image: " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
}

template <size_t CHANNELS>
bool Image<CHANNELS>::hasExtension(const std::string& filename, const std::string& targetExtension)
{
    size_t dotPos = filename.find_last_of('.');
    std::string fileExtension = filename.substr(dotPos + 1);

    return fileExtension.compare(targetExtension) == 0;
}

template <size_t CHANNELS>
bool Image<CHANNELS>::machineIsLittleEndian()
{
    int val = 1;
    std::byte *bytes = reinterpret_cast<std::byte*>(&val);

    return static_cast<bool>(bytes[0]);
}

template <size_t CHANNELS>
Array3 Image<CHANNELS>::reverseRGBByteOrder(Array3 rgb)
{
    for (int i = 0; i < 3; i++) {
        rgb[i] = reverseFloatByteOrder(rgb[i]);
    }

    return rgb;
}

template <size_t CHANNELS>
float Image<CHANNELS>::reverseFloatByteOrder(float val)
{
    float retVal;
    std::byte *bytes = reinterpret_cast<std::byte*>(&val);
    std::byte *retBytes = reinterpret_cast<std::byte*>(&retVal);
    retBytes[0] = bytes[3]; retBytes[1] = bytes[2];
    retBytes[2] = bytes[1]; retBytes[3] = bytes[0];

    return retVal;
}
