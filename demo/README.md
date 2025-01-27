# Zombie 2D Demo

To run the C++ demo, first build the Zombie library:

```bash
mkdir build
cd build && cmake ..
make -j4
```

Next you can run either [Walk on Stars](https://www.cs.cmu.edu/~kmcrane/Projects/WalkOnStars/index.html), [Boundary Value Caching](http://www.rohansawhney.io/BoundaryValueCaching.pdf) or [reverse WoSt](https://imaging.cs.cmu.edu/walk_on_stars_robin/) from the build directory:

```
./demo/demo ../demo/model_problems/engine/wost.json
./demo/demo ../demo/model_problems/engine/bvc.json
./demo/demo ../demo/model_problems/engine/rws.json
```

The results will be saved to `demo/model_problems/engine/solutions`.

To run the Python demo, use the following commands after building the Zombie bindings:

```
python demo/demo.py --config=demo/model_problems/engine/wost.json
python demo/demo.py --config=demo/model_problems/engine/bvc.json
python demo/demo.py --config=demo/model_problems/engine/rws.json
```

## Custom Model Problem Creation

The Zombie 2D demo allows custom model problems to be created by specifying a boundary geometry, a reflecting (Neumann or Robin) boundary mask, and boundary and source textures. The reflecting boundary indicator will determine whether a boundary is Dirichlet (black) or Neumann/Robin (white). The mapping from scene space to the mask is computed relative to the bounding box for the boundary geometry.

<div align='center'>
  <img src='./imgs/overview.png'/>
</div>


These model problem components are specified along with solver and output options via JSON files.

```
{
    "solverType": "wost",
    "solver": {
        "nWalks": 128,
        "maxWalkLength": 1024,
        "epsilonShellForAbsorbingBoundary": 1e-3,
        "epsilonShellForReflectingBoundary": 1e-3,
        "russianRouletteThreshold": 0.99,
        "ignoreAbsorbingBoundaryContribution": false,
        "ignoreReflectingBoundaryContribution": true,
        "ignoreSourceContribution": true
    },
    "modelProblem": {
        "geometry": "../demo/modelProblem/engine/data/geometry.obj",
        "isReflectingBoundary": "../demo/modelProblem/engine/data/is_reflecting_boundary.pfm",
        "absorbingBoundaryValue": "../demo/modelProblem/engine/data/absorbing_boundary_value.pfm",
        "reflectingBoundaryValue": "../demo/modelProblem/engine/data/reflecting_boundary_value.pfm",
        "sourceValue": "../demo/modelProblem/engine/data/source_value.pfm",
        "robinCoeff": 0.0,
        "absorptionCoeff": 0.0
    },
    "output": {
        "solutionFile": "../demo/modelProblem/engine/solutions/wost.pfm",
        "gridRes": 256,
        "boundaryDistanceMask": 1e-2,
        "saveDebug": false,
        "saveColormapped": true,
        "colormap": "turbo",
        "colormapMinVal": 0.0,
        "colormapMaxVal": 1.1
    }
}
```

You can use a 2D illustration tool to create geometry, a reflecting boundary indicator mask, and boundary and source textures. Assign each asset to a different layer and ensure that the size of the canvas matches the square bounding box for your boundary geometry as shown below

<div align='center'>
  <img src='./imgs/model_problem_builder.png'/>
</div>


Next export the geometry as an SVG and the reflecting boundary indicator and boundary and source textures as PNGs. You can then use the `svg2obj.py` script to convert the SVG to an OBJ.

```
./svg2obj.py engine/data/geometry.svg --normalize --auto_orient_curves
```

By changing the file extensions used in the config, you can use either PNGs or [PFMs](https://www.pauldebevec.com/Research/HDR/PFM/) for the texture data and solution outputs. Note that PNGs only support values between 0 and 255, which are clipped when writing out solutions whereas PFMs support arbitrary floating point values.
