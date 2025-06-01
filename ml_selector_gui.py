import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext, messagebox  # For a scrollable text area for history and displaying the About dialog

class MLAlgorithmSelectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Algorithm Selector")
        self.root.geometry("700x625")  # You can change this line for window size

        self.style = ttk.Style()
        try:
            self.style.theme_use('clam')  # Or 'alt', 'default', 'classic'
        except tk.TclError:
            print("Clam theme not available, using default.")

        # Styling
        self.style.configure("TLabel", padding=6, font=("Helvetica", 11))
        self.style.configure("TButton", padding=6, font=("Helvetica", 10, "bold"))
        self.style.configure("TRadiobutton", padding=(10, 5), font=("Helvetica", 10))
        self.style.configure("TCombobox", padding=5, font=("Helvetica", 10))
        self.style.configure("Title.TLabel", font=("Helvetica", 16, "bold"), foreground="#333333")
        self.style.configure("Question.TLabel", font=("Helvetica", 12, "italic"), foreground="#4F4F4F", wraplength=650)
        self.style.configure("Result.TLabel", font=("Helvetica", 13), foreground="#2E8B57", wraplength=650)
        self.style.configure("FinalResult.TLabel", font=("Helvetica", 14, "bold"), foreground="red", wraplength=650)
        self.style.configure("Error.TLabel", font=("Helvetica", 12, "bold"), foreground="#CC0000")
        self.style.configure("HistoryTitle.TLabel", font=("Helvetica", 12, "bold"), foreground="#333333")

        # Variables
        self.choice_var = tk.StringVar()
        self.history = []

        # Layout frames
        self.main_frame = ttk.Frame(root, padding="15 15 15 15")
        self.main_frame.pack(expand=True, fill=tk.BOTH)

        self.title_label = ttk.Label(self.main_frame, text="Machine Learning Algorithm Selector", style="Title.TLabel")
        self.title_label.pack(pady=(0, 20))
        
        # --- Menu ---
        menubar = tk.Menu(root)
        helpmenu = tk.Menu(menubar, tearoff=0)
        # helpmenu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="About", command=self.show_about)
        # menubar.add_cascade(label="Help", menu=helpmenu)
        root.config(menu=menubar)

        # History panel
        self.history_outer_frame = ttk.Frame(self.main_frame)
        self.history_outer_frame.pack(pady=(0, 15), fill=tk.X)
        self.history_title_label = ttk.Label(self.history_outer_frame, text="Your Path So Far:", style="HistoryTitle.TLabel")
        self.history_title_label.pack(anchor="w", pady=(0, 5))
        self.history_text_area = scrolledtext.ScrolledText(
            self.history_outer_frame, wrap=tk.WORD, height=8, font=("Courier New", 10),
            relief=tk.SOLID, borderwidth=1, bg="#f0f0f0", fg="#333333"
        )
        self.history_text_area.pack(fill=tk.X, expand=True)
        self.history_text_area.configure(state='disabled')

        # Question and options
        self.question_label = ttk.Label(self.main_frame, text="", style="Question.TLabel")
        self.question_label.pack(pady=(0, 15), anchor="w")
        self.options_frame = ttk.Frame(self.main_frame)
        self.options_frame.pack(pady=(0, 15), fill=tk.X, anchor="w")

        # Result label
        self.result_label = ttk.Label(self.main_frame, text="", style="Result.TLabel")
        self.result_label.pack(pady=(10, 15))

        # Navigation buttons
        self.nav_frame = ttk.Frame(self.main_frame)
        self.nav_frame.pack(pady=(20, 0), fill=tk.X, side=tk.BOTTOM, anchor='sw')
        self.submit_button = ttk.Button(self.nav_frame, text="Next", command=self.process_answer)
        self.submit_button.pack(side=tk.LEFT, padx=(0, 10))
        self.restart_button = ttk.Button(self.nav_frame, text="Restart", command=self.restart_quiz)
        self.restart_button.pack(side=tk.LEFT)

        # Load tree and initialize
        self.load_decision_tree()
        self.current_node_key = 'start'
        self.update_history_display()
        self.display_current_node()

    def load_decision_tree(self):
        # Enhanced decision tree with algorithm suggestions and practical tips
        self.decision_tree = {
            'start': {
                'text': "What is the primary type of task you want to perform?",
                'type': 'choice',
                'options': {
                    "Predict a continuous value (Regression)": 'regression_start',
                    "Classify data into categories": 'classification_start',
                    "Forecast future values (Time Series)": 'forecast_start',
                    "Group or segment unlabeled data (Clustering)": 'clustering_start',
                    "Reduce data dimensionality": 'dimensionality_reduction_start',
                    "Generate synthetic data (Generative AI)": 'generative_ai_start',
                    "Special Classification Scenarios": 'special_classification_start'
                }
            },
            # --- REGRESSION ---
            'regression_start': {
                'text': "Is the relationship between inputs and output likely linear?",
                'type': 'yes_no',
                'next_yes': 'regression_linear_multicollinearity',
                'next_no': 'regression_nonlinear_complex'
            },
            'regression_linear_multicollinearity': {
                'text': "Are there many features with potential multicollinearity?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['Ridge Regression', 'Lasso Regression'],
                        'tips': 'Standardize features; use cross-validated regularization parameter α; inspect coefficient paths to choose between L2 and L1 penalties.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['Ordinary Least Squares (Linear Regression)'],
                        'tips': 'Center and scale inputs; check residual plots and VIF; add interaction terms only with domain justification.'
                    }
                }
            },
            'regression_nonlinear_complex': {
                'text': "Are there likely nonlinear or complex patterns?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['Random Forest', 'XGBoost', 'LightGBM', 'Polynomial Regression (deg≤3) + Regularization', 'Neural Networks'],
                        'tips': 'Use ensemble trees for automatic feature interactions; limit polynomial degree to ≤3; apply grid-search with SHAP for interpretability.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['Elastic Net', 'k-NN Regression (k≈√n)'],
                        'tips': 'For Elastic Net combine L1+L2 to balance selection and shrinkage; scale features when using k-NN.'
                    }
                }
            },
            # --- CLASSIFICATION ---
            'classification_start': {
                'text': "Is the output binary or multi-class?",
                'type': 'choice',
                'options': {
                    "Binary": 'classification_binary',
                    "Multi-class": 'classification_multiclass'
                }
            },
            'classification_binary': {
                'text': "Are the classes mostly linearly separable?",
                'type': 'yes_no',
                'next_yes': 'classification_binary_interpret',
                'next_no': 'classification_binary_nonlinear'
            },
            'classification_binary_interpret': {
                'text': "Is interpretability critical?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['Logistic Regression (L2)'],
                        'tips': 'Inspect coefficients; apply coefficient significance tests; standardize inputs.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['Linear SVM', 'k-NN (tune k via CV)', 'Naive Bayes'],
                        'tips': 'Use Platt scaling or isotonic regression if probabilistic outputs needed.'
                    }
                }
            },
            'classification_binary_nonlinear': {
                'text': "Do you need probability estimates?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['Calibrated SVM', 'Neural Network with Softmax'],
                        'tips': 'Use cross-validation for calibration; consider Platt scaling.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['Random Forest', 'XGBoost', 'SVM (RBF)'],
                        'tips': 'Tune tree depth and learning rate; monitor ROC-AUC.'
                    }
                }
            },
            'classification_multiclass': {
                'text': "Are classes mutually exclusive or multi-label?",
                'type': 'choice',
                'options': {
                    "Exclusive": 'classification_multiclass_excl',
                    "Multi-label": 'classification_multilabel'
                }
            },
            'classification_multiclass_excl': {
                'text': "Prioritize simplicity/efficiency or performance?",
                'type': 'choice',
                'options': {
                    "Simplicity/Efficiency": {
                        'result': {
                            'algorithms': ['Multinomial Logistic Regression', 'Naive Bayes'],
                            'tips': 'Good for high-dimensional sparse data; fast training.'
                        }
                    },
                    "Performance": {
                        'result': {
                            'algorithms': ['One-vs-One SVM', 'Random Forest', 'XGBoost', 'Neural Networks'],
                            'tips': 'Ensure balanced classes; use ensemble stacking if needed.'
                        }
                    }
                }
            },
            'classification_multilabel': {
                'result': {
                    'algorithms': ['Binary Relevance', 'Classifier Chains', 'Neural Networks for Multi-label'],
                    'tips': 'Parallelize label-wise models; use label correlation methods for dependent labels.'
                }
            },
            # --- TIME SERIES FORECASTING ---
            'forecast_start': {
                'text': "Short-term or long-term forecasting?",
                'type': 'choice',
                'options': {
                    "Short-term": 'forecast_short',
                    "Long-term": 'forecast_long'
                }
            },
            'forecast_short': {
                'result': {
                    'algorithms': ['ARIMA/SARIMA', 'Holt–Winters', 'Prophet'],
                    'tips': 'Use AIC/BIC for order selection; incorporate external regressors if available.'
                }
            },
            'forecast_long': {
                'text': "Sufficient data for deep learning?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['LSTM', 'GRU', 'Temporal CNN'],
                        'tips': 'Use rolling-window cross-validation; normalize sequences.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['XGBoost on lag features', 'VAR', 'Theta Model'],
                        'tips': 'Engineer seasonality and lag features; tune tree-based parameters.'
                    }
                }
            },
            # --- CLUSTERING ---
            'clustering_start': {
                'text': "Expect spherical clusters of similar size?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['k-Means', 'k-Medoids'],
                        'tips': 'Use elbow and silhouette scores to choose k.'
                    }
                },
                'next_no': 'clustering_complex'
            },
            'clustering_complex': {
                'text': "Clusters with arbitrary shape or density?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['DBSCAN', 'HDBSCAN', 'OPTICS'],
                        'tips': 'Tune epsilon/min_samples; visualize reachability plots.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['Agglomerative Clustering', 'Spectral Clustering', 'Affinity Propagation'],
                        'tips': 'Use dendrograms to inspect cluster hierarchies.'
                    }
                }
            },
            # --- DIMENSIONALITY REDUCTION ---
            'dimensionality_reduction_start': {
                'text': "Is primary goal visualization?",
                'type': 'yes_no',
                'next_yes': 'dim_reduce_viz',
                'next_no': {
                    'result': {
                        'algorithms': ['Autoencoders', 'Factor Analysis', 'Manifold Learning'],
                        'tips': 'Choose latent dim ≈5-10% of input dims; standardize inputs.'
                    }
                }
            },
            'dim_reduce_viz': {
                'text': "Preserve global variance or local structure?",
                'type': 'yes_no',
                'next_yes': {
                    'result': {
                        'algorithms': ['PCA'],
                        'tips': 'Generate scree plot; select components covering 90% variance.'
                    }
                },
                'next_no': {
                    'result': {
                        'algorithms': ['t-SNE', 'UMAP'],
                        'tips': 'UMAP for speed; t-SNE for fine-grained clusters; tune perplexity or n_neighbors.'
                    }
                }
            },
            # --- GENERATIVE AI ---
            'generative_ai_start': {
                'text': "What data type to generate?",
                'type': 'choice',
                'options': {
                    "Image": {
                        'result': {
                            'algorithms': ['GANs (StyleGAN2)', 'VAEs', 'Diffusion Models'],
                            'tips': 'Fine-tune on <1k images; use spectral normalization.'
                        }
                    },
                    "Text": {
                        'result': {
                            'algorithms': ['GPT-family'],
                            'tips': 'Apply prompt/template tuning; add safety filters.'
                        }
                    },
                    "Tabular": {
                        'result': {
                            'algorithms': ['CTGAN', 'TVAE'],
                            'tips': 'Validate marginals; enforce feature constraints.'
                        }
                    },
                    "Audio": {
                        'result': {
                            'algorithms': ['WaveGAN', 'HiFi-GAN'],
                            'tips': 'Train on spectrograms; upsample for quality.'
                        }
                    },
                    "Video": {
                        'result': {
                            'algorithms': ['Temporal GANs', '3D CNN'],
                            'tips': 'Enforce temporal consistency; use 3D discriminators.'
                        }
                    }
                }
            },
            # --- SPECIAL CLASSIFICATION SCENARIOS ---
            'special_classification_start': {
                'text': "Which special scenario applies?",
                'type': 'choice',
                'options': {
                    "Small dataset": {
                        'result': {
                            'algorithms': ['k-NN', 'Naive Bayes', 'Logistic Regression'],
                            'tips': 'Use data augmentation and bootstrapped CV.'
                        }
                    },
                    "Missing values": {
                        'result': {
                            'algorithms': ['LightGBM', 'XGBoost'],
                            'tips': 'Handle natively or impute with IterativeImputer.'
                        }
                    },
                    "Imbalanced classes": 'imbalanced_class',
                    "High-dimensional": 'high_dimensional',
                    "Real-time needs": {
                        'result': {
                            'algorithms': ['Logistic Regression', 'Linear SVM'],
                            'tips': 'Ensure model size and latency meet requirements.'
                        }
                    },
                    "Noisy data": {
                        'result': {
                            'algorithms': ['RANSAC', 'Isolation Forest'],
                            'tips': 'Robust to outliers; use median-based ensembles.'
                        }
                    }
                }
            },
            'imbalanced_class': {
                'text': "Severity of imbalance?",
                'type': 'choice',
                'options': {
                    "Moderate (1:5–1:10)": {
                        'result': {
                            'algorithms': ['Weighted Random Forest', 'Cost-sensitive SVM'],
                            'tips': 'Use stratified sampling; adjust class weights.'
                        }
                    },
                    "Severe (>1:10)": {
                        'result': {
                            'algorithms': ['SMOTE + Ensemble', 'Focal Loss in NN'],
                            'tips': 'Combine oversampling with stacking; tune focal loss gamma.'
                        }
                    }
                }
            },
            'high_dimensional': {
                'text': "Primary concern with many features?",
                'type': 'choice',
                'options': {
                    "Feature selection": {
                        'result': {
                            'algorithms': ['Lasso', 'RFE', 'RF feature importance'],
                            'tips': 'Use recursive elimination and cross-validation.'
                        }
                    },
                    "Curse of dimensionality": {
                        'result': {
                            'algorithms': ['Elastic Net', 'Dim Reduction + Classifier'],
                            'tips': 'Apply PCA before classification; use dropout in NNs.'
                        }
                    }
                }
            }
        }

    def format_recommendation(self, rec_dict):
        """
        Takes a recommendation dict with keys 'algorithms' (list) and 'tips' (string),
        and returns a nicely formatted multi-line string.
        """
        algorithms_list = rec_dict.get('algorithms', [])
        tips_text = rec_dict.get('tips', '')

        # Wrap each algorithm name in single quotes
        algorithms_str = ", ".join(f"'{alg}'" for alg in algorithms_list)
        # Format as two lines
        formatted = f"1. Algorithms: {algorithms_str}\n" \
                    f"2. Tips: '{tips_text}'"
        return formatted

    def update_history_display(self):
        self.history_text_area.configure(state='normal')
        self.history_text_area.delete(1.0, tk.END)
        if not self.history:
            self.history_text_area.insert(tk.END, "No selections made yet.\nYour choices will appear here.\n")
        else:
            for i, (question, answer) in enumerate(self.history):
                self.history_text_area.insert(tk.END, f"{i+1}. Q: {question}\n   A: {answer}\n\n")
        self.history_text_area.configure(state='disabled')
        self.history_text_area.see(tk.END)  # Auto-scroll

    def display_current_node(self):
        # Clear previous options
        for widget in self.options_frame.winfo_children():
            widget.destroy()

        # Clear previous result text
        self.result_label.config(text="", style="Result.TLabel")
        self.submit_button.config(state=tk.NORMAL, text="Next")

        node = self.decision_tree.get(self.current_node_key)
        if not node:
            self.question_label.config(text="Error: Node not found!", style="Error.TLabel")
            self.submit_button.config(state=tk.DISABLED)
            return

        # If leaf (result node)
        if 'result' in node:
            # Display leaf text (if any) or a default message
            self.question_label.config(text=node.get('text', "Path Complete!"))
            # Format the recommendation dictionary
            formatted = self.format_recommendation(node['result'])
            self.result_label.config(text=f"Recommendation:\n{formatted}", style="FinalResult.TLabel")
            self.submit_button.config(state=tk.DISABLED, text="Done")
            return

        # Not a leaf: display question
        self.question_label.config(text=node['text'])

        # Handle yes/no questions
        if node['type'] == 'yes_no':
            self.choice_var.set('yes')
            ttk.Radiobutton(self.options_frame, text="Yes", variable=self.choice_var, value="yes").pack(anchor="w", pady=3)
            ttk.Radiobutton(self.options_frame, text="No", variable=self.choice_var, value="no").pack(anchor="w", pady=3)

        # Handle multiple-choice questions
        elif node['type'] == 'choice':
            options = list(node['options'].keys())
            self.choice_var.set(options[0])
            combobox = ttk.Combobox(
                self.options_frame,
                textvariable=self.choice_var,
                values=options,
                state="readonly",
                width=60
            )
            combobox.pack(anchor="w", pady=3, padx=5, fill=tk.X)

        self.options_frame.update_idletasks()

    def process_answer(self):
        node = self.decision_tree.get(self.current_node_key)
        if not node or 'result' in node:
            return

        user_answer_val = self.choice_var.get()
        question_text = node['text']

        # Determine display_answer for history
        if node['type'] == 'yes_no':
            display_answer = "Yes" if user_answer_val == 'yes' else "No"
        else:
            display_answer = user_answer_val

        # Append to history and update display
        self.history.append((question_text, display_answer))
        self.update_history_display()

        # Determine next node or result
        if node['type'] == 'yes_no':
            next_info = node['next_yes'] if user_answer_val == 'yes' else node['next_no']
        else:  # 'choice'
            next_info = node['options'].get(user_answer_val)

        # If next_info is a string, it's the next node key
        if isinstance(next_info, str):
            self.current_node_key = next_info

        # If next_info is a dict with 'result', create a temporary leaf
        elif isinstance(next_info, dict) and 'result' in next_info:
            temp_key = f"result_leaf_{self.current_node_key}_{''.join(filter(str.isalnum, user_answer_val))}"
            idx = 1
            # Ensure unique temp_key
            while temp_key in self.decision_tree:
                temp_key = f"{temp_key}_{idx}"
                idx += 1
            # Insert temporary node with its result
            self.decision_tree[temp_key] = {
                'text': f"Result for: {display_answer}",
                'result': next_info['result']
            }
            self.current_node_key = temp_key

        else:
            # Undefined path
            self.result_label.config(text=f"Error: Path undefined for '{user_answer_val}'", style="Error.TLabel")
            self.submit_button.config(state=tk.DISABLED)
            # Remove the last history entry
            if self.history:
                self.history.pop()
            self.update_history_display()
            return

        # Display the next node or result
        self.display_current_node()

    def restart_quiz(self):
        self.current_node_key = 'start'
        self.history.clear()
        # Remove any temporary leaves created
        for k in list(self.decision_tree.keys()):
            if k.startswith("result_leaf_"):
                del self.decision_tree[k]
        self.update_history_display()
        self.display_current_node()
        self.submit_button.config(state=tk.NORMAL, text="Next")

    def show_about(self):
        about_text = """
ML Algorithm Selector GUI

An educational tool to guide users in choosing the most suitable machine learning model, along with practical tips for each option.

By Claudio A. Gelmi (2025)  @github.com/cagelmi

Inspired by the article "Choosing the Right Machine Learning Algorithm: A Decision Tree Approach", by Iván Palomares Carrascosa (Kdnuggets May 21, 2025)
        """
        messagebox.showinfo("About", about_text)


if __name__ == "__main__":
    root = tk.Tk()
    app = MLAlgorithmSelectorGUI(root)
    root.mainloop()