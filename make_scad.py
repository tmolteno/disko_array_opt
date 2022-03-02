import json
import argparse
import numpy as np


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Generate a CAD file from output', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--json', default='optimized_array.json', help="input JSON file of antenna positions.")
    parser.add_argument('--output', default='optimized_array.scad', help="Output SCAD file.")

    ARGS = parser.parse_args()

    with open(ARGS.json, 'r') as infile:
        data = json.load(infile)

    arms = data['arms']
    angles = data['arm_degrees']
    
    with open(ARGS.output, 'w') as f:
        print('// Generated SCAD file for optimized TART array.', file=f)
        print('// Tim Molteno tim@elec.ac.nz.', file=f)
        print(f"// C/N = {data['C/N'] :4.2f}.", file=f)
        print(f"// Penalty = {data['penalty'] :4.2f}.", file=f)
        print(f"//", file=f)
    
        print('module antenna() { color("black") cylinder(r=25, h=50); }', file=f)
        
        print("module arm() { translate([0,-50, -50]) cube(size=[2000, 100, 50]); }", file=f)
        print("post_height = 1800;", file=f)
        print("module post() {cylinder(r=100, h=post_height);}", file=f)

        print("post();", file=f)
        print("translate([0,0,post_height]) {", file=f)

        
        for angle, arm in zip(angles, arms):
            armlist = np.array2string(np.array(arm), formatter={'float_kind':lambda x: "%.3f, " % x})

            print(f"for (r = {armlist}) rotate({angle}) {{ translate([r*1000, 0,0]) antenna(); }}", file=f)
            print(f"""rotate({angle}) arm();""", file=f)
        
        print("}", file=f)
