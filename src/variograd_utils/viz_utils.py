# colormaps

import matplotlib.colors



def Plot_to_numpy(plot):
    """
    Convert a surfplot.Plot object to a numpy array for including it in complex plottings
    (e.g. in matplotlib.pyplot subplots).

    Parameters
    ----------
    plot : surfplot.Plot
        The plot object to be converted to a numpy array.

    Returns
    -------
    numpy.ndarray
        The plot as a numpy array.

    Notes
    -----
    This function is a workaround for the fact that surfplot.Plot objects cannot be directly
    included in matplotlib.pyplot subplots.

    """

    plot = plot.render()
    plot._check_offscreen()
    return plot.to_numpy()

pearl1 = matplotlib.colors.LinearSegmentedColormap.from_list("", [(.361, .408, .596), # purple
                                                                (.920, .835, .980),  # lilac
                                                                (1, 1, .98),        # white
                                                                (.663, .875, .886), # teal
                                                                (.050, .192, .227), # darkteal
                                                                ])

pearl1_inv = matplotlib.colors.LinearSegmentedColormap.from_list("", [(1, 1, .98),        # white
                                                                      (.920, .835, .980), # lilac
                                                                      (.361, .408, .596), # purple
                                                                      (.0, .08, .20),     # rich black
                                                                      (.0150, .292, .327), # darkteal
                                                                      (.663, .875, .886), # teal
                                                                      (1, 1, .98),        # white
                                                                      ])



pearl2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FCF6FB", #whiteish
                                                                  "#EFC8E5", #turquoise
                                                                  "#EFC8E5",
                                                                  "#E4C7EB",
                                                                  "#D9B9E9",
                                                                  "#D2B3E5", #lilac
                                                                  "#CCCCF0",
                                                                  "#C4DCED",
                                                                  "#C4EDED",
                                                                  "#C4EDED", #pink
                                                                  "#F6FCFC"] #whiteish
                                                                  )



sunset1 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#264653", # rich black
                                                                   "#005F73", # midnight green
                                                                   "#0A9396", # dark cyan
                                                                   "#94D2BD", # tiffany blue
                                                                   "#E9D8A6", # vanilla
                                                                   "#E9D8A6", # vanilla
                                                                   "#EE9B00", # gamboge
                                                                   "#CE5E09", #tawny
                                                                   "#8C180D", # dark red
                                                                   "#450E0F"  # black bean
                                                                   ])

sunset1_inv = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#E9D8A6", # vanilla
                                                                   "#94D2BD", # tiffany blue
                                                                   "#0A9396", # dark cyan
                                                                   "#005F73", # midnight green
                                                                   "#264653", # rich black
                                                                   "#000000", # black
                                                                   "#450E0F",  # black bean
                                                                   "#8C180D", # dark red
                                                                   "#CE5E09", #tawny
                                                                   "#EE9B00", # gamboge
                                                                   "#E9D8A6" # vanilla
                                                                   ])


sandy = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FEB1A3", # melon
                                                                   "#F7CDC6", # tpale dogwood
                                                                   "#F8EDEB", # misty rose
                                                                   "#E1E1D3", # alabaster
                                                                   "#C9DCD1", # ash gray
                                                                   "#C9DCD1",  # ash gray
                                                                   "#DADCCE", # alabaster
                                                                   "#EBDBCA", # almond
                                                                   "#F3C59B", # peach
                                                                   "#FAAE6B" # sandy brown
                                                                   ])


pidgeon = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#BB064C",
                                                                    "#A51357",
                                                                    "#932065",
                                                                    "#812C7E",
                                                                    "#563993",
                                                                    "#335899",
                                                                    "#2272A0",
                                                                    "#1184A7",
                                                                    "#0090AD",
                                                                    ])


pastelbow = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFADAD", # melon
                                                                    "#FFD6A5", # subset
                                                                    "#FDFFB6", # cream
                                                                    "#CAFFBF", # tea green
                                                                    "#B3FBDF", # aquamarine
                                                                    "#9BF6FF", # electric blue
                                                                    "#9EDDFF", # pale azure
                                                                    "#A0C4FF", # jordy blue
                                                                    "#BDB2FF", # periwinkle
                                                                    "#FFC6FF" # mauve
                                                                    ])


wisteria = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#2E3431", 
                                                                    "#3C6C39", # yellow green
                                                                    "#A7C957", 
                                                                    "#F3FE88",  
                                                                    "#F4F4F4", # ivory
                                                                    "#DABFFF", 
                                                                    "#8C65AF",  
                                                                    "#3E0B5E", # dark purple
                                                                    "#2F2637"
                                                                    ])

wisteria_inv = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#F4F6E7", # ivory
                                                                        "#F3FE88", 
                                                                        "#A7C957", 
                                                                        "#3C6C39", # yellow green
                                                                        "#2C2B2F",
                                                                        "#3E0B5E", # dark purple
                                                                        "#8C65AF", 
                                                                        "#DABFFF",
                                                                        "#F1EEF6"  # magnolia
                                                                    ])


kelpflare = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#1E3019",
                                                                     "#394E31",
                                                                     "#EEFEEE",
                                                                     "#E8D0D8",
                                                                     "#997982"
                                                                    ])


# coldhot = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#001F3D",
#                                                                    "#69BBD8",
#                                                                    "#C3E1EA",
#                                                                    "#F4F4F6",
#                                                                    "#F6C6C8",
#                                                                    "#F48B91",
#                                                                    "#CE123B",
#                                                                     ])
coldhot = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#001F3D",
                                                                   "#356D8B",
                                                                   "#356D8B",
                                                                   "#ADD8E6",
                                                                   "#EBE9ED",
                                                                   "#F1BBBD",
                                                                   "#E0677C",
                                                                   "#CE123B",
                                                                   "#8A0A36",
                                                                    ])


coldhot_inv = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#F4F4F6",
                                                                       "#F6C6C8",
                                                                       "#EB5069",
                                                                       "#A90F30",
                                                                       "#121216",
                                                                       "#003366",
                                                                       "#69BBD8",
                                                                       "#C3E1EA",
                                                                       "#F4F4F6"
                                                                    ])



heucera = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#5D3251",
                                                                   "#854773",
                                                                   "#CCA8D8",
                                                                   "#F2F5F8",
                                                                   "#6EC6CA",
                                                                   "#078688",
                                                                   "#03393A"
                                                                    ])
heucera_inv = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#F2F5F8",
                                                                       "#6EC6CA",
                                                                       "#078688",
                                                                       "#03393A",
                                                                       "#191C24",
                                                                       "#5D3251",
                                                                       "#854773",
                                                                       "#CCA8D8", 
                                                                       "#F2F5F8", 
                                                                       ])

quitefire = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFF5F7", 
                                                                     "#FFD8DF",
                                                                     "#FFC2CC",
                                                                     "#FFB2AB",
                                                                     "#FFA28A",
                                                                     "#FF8247",
                                                                     "#B11F08",
                                                                     "#3A0A03",
                                                                     "#111111"
                                                                    ])
quitefire = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#FFF5F7", 
                                                                     "#FFC2CC",
                                                                     "#FFB2AB",
                                                                     "#FFA28A",
                                                                     "#FF8247",
                                                                     "#D85128",
                                                                     "#B11F08",
                                                                     "#130301"
                                                                    ])

pond = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#F4F4F4",
                                                                "#F3FE88",
                                                                "#A7C957",
                                                                "#729B48",
                                                                "#3D9168",
                                                                "#078688",
                                                                "#001F3D",
                                                               ])


bowrain = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#E9F39A",
                                                                   "#5AD7AB",
                                                                   "#30D9D3",
                                                                   "#0C83AB",
                                                                   "#4B4EAA",
                                                                   "#9577BB",
                                                                   "#DFA0CC",
                                                                   "#EF476F",
                                                                   "#F9A48B"
                                                                   ])