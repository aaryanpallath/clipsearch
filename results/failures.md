# Where zero-shot CLIP misses

Test queries whose correct image was not CLIP's first result. Links open the COCO images.

## counting

- **"A couple of people that are standing near a train."** (correct image ranked 13): [wanted](http://images.cocodataset.org/val2017/000000566524.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000049060.jpg), described as "A train sitting next to a loading platform with people standing on it."
- **"A plate with two slices of pizza on it."** (correct image ranked 7): [wanted](http://images.cocodataset.org/val2017/000000294831.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000337498.jpg), described as "A large slice of pizza with cheese and marinara sauce on a plate."
- **"A woman eating a sandwich in front of two paper cups filled with a hot liquid."** (correct image ranked 3): [wanted](http://images.cocodataset.org/val2017/000000157601.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000137246.jpg), described as "A woman holding a hot dog on the street."
- **"two elephants standing in some bushes beside a trail"** (correct image ranked 3): [wanted](http://images.cocodataset.org/val2017/000000097230.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000080274.jpg), described as "Two elephants that are standing next to each other."

## spatial

- **"A large clock on the side of a building above cars on the street."** (correct image ranked 2): [wanted](http://images.cocodataset.org/val2017/000000191013.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000150930.jpg), described as "The clock shown above has someone's name on it."
- **"A toilet right next to a shower stall."** (correct image ranked 5): [wanted](http://images.cocodataset.org/val2017/000000234413.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000175387.jpg), described as "Bathroom stall with green door in commercial business."
- **"The cars are travelling behind the guy on the bicycle"** (correct image ranked 5): [wanted](http://images.cocodataset.org/val2017/000000122166.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000169996.jpg), described as "Men on bikes riding alongside a car on the street"
- **"A chocolate frosted doughnut sitting on top of a plate."** (correct image ranked 2): [wanted](http://images.cocodataset.org/val2017/000000255824.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000035682.jpg), described as "Eating a donut makes for a quick and easy breakfast."

## negation

- **"Somebody is in the photograph not certain who that individual is."** (correct image ranked 398): [wanted](http://images.cocodataset.org/val2017/000000067315.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000002299.jpg), described as "Many small children are posing together in the black and white photo."
- **"Empty stylish kitchen and living room with big picture windows."** (correct image ranked 5): [wanted](http://images.cocodataset.org/val2017/000000472046.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000322895.jpg), described as "A living room that has a bunch of different couches."
- **"A picture of some type of park with benches and no people around."** (correct image ranked 4): [wanted](http://images.cocodataset.org/val2017/000000078032.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000551350.jpg), described as "The picnic table under the tree is empty of people."
- **"There is no image here to provide a caption for."** (correct image ranked 113): [wanted](http://images.cocodataset.org/val2017/000000157365.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000024243.jpg), described as "The girl is sitting inside the broken fridge on the side of the road."

## text in image

- **"Advertising sign above a patio umbrella on sunny day."** (correct image ranked 5): [wanted](http://images.cocodataset.org/val2017/000000341058.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000416104.jpg), described as "A group of people sitting under umbrellas drinking beer."
- **"Two signs that are on a pole in the snow."** (correct image ranked 2): [wanted](http://images.cocodataset.org/val2017/000000101022.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000366178.jpg), described as "a road that has some snow all around it"
- **"a building and some signs and street light and phone lines"** (correct image ranked 28): [wanted](http://images.cocodataset.org/val2017/000000404601.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000261796.jpg), described as "A black and white picture of a large house is shown."
- **"a road that has a small red stop sign by it"** (correct image ranked 10): [wanted](http://images.cocodataset.org/val2017/000000193884.jpg), [CLIP's first result](http://images.cocodataset.org/val2017/000000565153.jpg), described as "a road sign showing to stop on the road near a forest"
