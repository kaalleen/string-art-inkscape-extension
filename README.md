## Inkscape Extension: Convert Image to String Art

Converts an image into string art.

Original code is taken from
[https://github.com/kaspar98/StringArt/](https://github.com/kaspar98/StringArt/)
and has been converted to an Inkscape extension.

### How To Install

Download and extract to your Inkscape extensions folder.

### How To Use The Extension

* Select an image
* Open `Extensions > Image > Convert Image to String Art`
* Enter prefered settings
* Apply

### Settings

* Thread width: defines thread width (during caclculation) and the output stroke width
* Number lines (per color): defines how many lines will be generated. For multiple colors this value
  will apply to each color. A value of 0 will generate as many lines until no improvement is
  detectable.
* Number od random nails to pick from for the next iteration: number of random nails to pick from
  when choosing the next nail to speed up the algorithm at the cost of quality.
  A value of 0 will calculate values for all possible nails at every iteration.
  A good value for this is ~50.
* Approximate nail distance: Defines how many nails will be used around the image. Smaller values
  lead to a larger amount of nails.
* White on black: Inverts image colors
* Shape: defines the output shape. Possible options are circle, ellipse or rectangle.
* Color Mode: Single color or Multicolor (RGB)
* Insert Nail Numbers: Wether to output nail numbers around the image (every 5th)
* Output Nail Order: Wether to output a list of nail numbers to use while creating
  the actual artwork
