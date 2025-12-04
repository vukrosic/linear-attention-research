1. not enough compute, just show theory and concepts, train a bit
2. show wich tokens get routed where in the inference
3. maybe also train full softmax, show all 3 training curves

This will allow you to empirically verify your hypothesis:

Do "easy" words like "The", "a", "of" go to Linear?
Do "hard" words or entities like "quantum", "mechanics" go to Softmax?


How to use it
Train the model first (if you haven't already):
bash
python run_experiment.py --config dynamic
Run the visualization:
bash
python visualize_routing.py --text "The quick brown fox jumps over the lazy dog. However, the complex nuances of quantum mechanics require deeper understanding."
What it does
It runs the model on your text.
It prints a table in the terminal showing each token and whether it went to LINEAR (Blue) or SOFTMAX (Red) for Layer 1 and Layer 2.
It also saves an HTML file (routing_viz.html) that you can open in a browser for a cleaner view.