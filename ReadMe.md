# DLO Approximation

In deformable linear object (DLO) control applications the object is usually represented as a set of linear segments. More segments provide better approximation of the object, but the number of segments increases the computational cost of the control methods. Especially, when planning a path for the DLO as the number of segments increases, the path planning becomes computationally expensive. 

## What does this project provide?
In this project, we provide a fast method to approximate the DLO with a fewer number of segments given an original DLO representation. 

## Advantages:
The main advantage of this method is that, unlike classical the interpolation techniques, this method approximates the shape of the DLO as closely as possible to the original DLO representation with **equal length segments** while **preserving the total length** of the DLO. 

Note that, an interpolation based method would be a poor approximation for DLOs with only a few segments. Consider the case with only one segment approximation with interpolation for the example DLO states in the Examples section below. An interpolation method would result in a line segment from the first point to the last point which neither maintains the total length of the DLO nor preserves the main position and orientation distribution of the DLO.

Moreover, the method is **computationally  efficient** as opposed to an optimization based fitting method. An optimization based fitting method would require the optimization of the objective function which is a non-linear function of the DLO state. 

## Method 
### Input parameters:

The method takes the following parameters:
- `l_dlo`: Length of the original DLO
- `dlo_state`: State of the original DLO as a list of points (x, y, z) and quaternions (x, y, z, w)
- `num_seg_d`: Desired number of segments to approximate the DLO with. (between 1 to the length of `dlo_state`)

### Output parameters:

The method returns 
- `approximated_pos`: The approximated DLO state as a list of points (x, y, z) with length equal to `num_seg_d + 1`. The first point is the starting point of the first segment and the last point is the ending point of the last segment.
- `max_angle`: The maximum rotation angle between the approximated line segments. Useful for making sure the aproximation does not introduce large rotations between the segments.
- `avg_error`: The average distance error between the original and approximated positions per original segment. 

### Method implementation:

The method is implemented in `dlo_state_approximator.py`. 


## Example Approximations:

To demonstrate the method, we use two examples states of a DLO with 0.5 meter length and 40 segments as shown below with the screen shots taken from RVIZ. 

Example 1                |  Example 2
:-------------------------:|:-------------------------:
![](./.imgs/dlo_state_example_1_cropped.png)  |  ![](./.imgs/dlo_state_example_2_cropped.png)


The states of the DLOs are extrated to CSV files (given in `dlo_state_example_1.csv` and `dlo_state_example_2.csv` files) and can be loaded using Python. An example is provided in `dlo_state_csv_plotter.py` generates the plots shown below.
The center points of the original DLO segments are shown with red points and directions of the DLO segments are shown with blue arrows.

Example 1                |  Example 2
:-------------------------:|:-------------------------:
![](./.imgs/reading_from_csv_example_1.png)  |  ![](./.imgs/reading_from_csv_example_2.png)

Next, their approximated representations with 1, 2, 3, and 4 segments can be obtained using the function in `dlo_state_csv_approximator.py` script. The center points of the original DLO segments are shown with blue points and the approximated DLO segments are shown in orange lines.  

Example 1                |  Example 2
:-------------------------:|:-------------------------:
1 Segment Approximation  |  1 Segment Approximation
![](./.imgs/approx_ex1_num_seg_01.png)  |  ![](./.imgs/approx_ex2_num_seg_01.png)
2 Segment Approximation  |  2 Segment Approximation
![](./.imgs/approx_ex1_num_seg_02.png)  |  ![](./.imgs/approx_ex2_num_seg_02.png)
3 Segment Approximation  |  3 Segment Approximation
![](./.imgs/approx_ex1_num_seg_03.png)  |  ![](./.imgs/approx_ex2_num_seg_03.png)
4 Segment Approximation  |  4 Segment Approximation
![](./.imgs/approx_ex1_num_seg_04.png)  |  ![](./.imgs/approx_ex2_num_seg_04.png)

Although it is not always a guaranteed behavior, in general as the number of segments increases, the approximation becomes more accurate as shown in the approximations with 5, 10, 20, and 40 segments.

Example 1                |  Example 2
:-------------------------:|:-------------------------:
5 Segment Approximation  |  5 Segment Approximation
![](./.imgs/approx_ex1_num_seg_05.png)  |  ![](./.imgs/approx_ex2_num_seg_05.png)
10 Segment Approximation  |  10 Segment Approximation
![](./.imgs/approx_ex1_num_seg_10.png)  |  ![](./.imgs/approx_ex2_num_seg_10.png)
20 Segment Approximation  |  20 Segment Approximation
![](./.imgs/approx_ex1_num_seg_20.png)  |  ![](./.imgs/approx_ex2_num_seg_20.png)
40 Segment Approximation  |  40 Segment Approximation
![](./.imgs/approx_ex1_num_seg_40.png)  |  ![](./.imgs/approx_ex2_num_seg_40.png)


### Average distance error wrt number of segments
The average distance error between the original and approximated positions per original segment with respect to the number of segments are plotted as shown below.


Example 1                |  Example 2
:-------------------------:|:-------------------------:
TODO  |  TODO

### Maximum rotation angle wrt number of segments
The maximum rotation angle between the approximated line segments with respect to the number of segments are plotted as shown below.

Example 1                |  Example 2
:-------------------------:|:-------------------------:
TODO  |  TODO

### Run times wrt number of segments
The run times of the method with respect to the number of segments are plotted as shown below. (Measured on Intel(R) Core(TM) i9-10885h CPU @ 2.40GHz)

Example 1                |  Example 2
:-------------------------:|:-------------------------:
TODO  |  TODO