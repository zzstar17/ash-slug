# Ash-slug

Library dedicated to process text data for use in [Lengyel's Slug font rendering algorithm](https://github.com/EricLengyel/Slug)
released shader implementations. It assists with populating vertex/index buffers and textures used in the shaders.

Although this library was designed for use with [Ash](https://github.com/ash-rs/ash), a wrapper around Vulkan, the Ash bindings
are actually optional and can be designed by not enabling the "ash" feature (which is enabled by default). The text processing
code is available for use in other Vulkan wrappers and graphics APIs.

This library depends on [HarfRust](https://github.com/harfbuzz/harfrust) for text shaping and
[ttf-parser](https://github.com/harfbuzz/ttf-parser) for parsing fonts.

## Vulkan shaders and changes

The Vulkan version of shaders are available in this repository.
The initial version is also available [in the original reference implementation repository](https://github.com/EricLengyel/Slug).

## More information

See https://terathon.com/blog/decade-slug.html

## Acknowledgements

Thank you to diffusionstudio for [providing the initial inspiration for the library](https://github.com/diffusionstudio/slug-webgpu)
and of course a big thank you to Eric Lengyel for creating the Slug algorithm and releasing it into the public domain.
