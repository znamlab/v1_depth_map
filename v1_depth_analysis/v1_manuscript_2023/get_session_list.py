import flexiznam as flz


def get_sessions(project, mouse_list, exclude_sessions=[]):
    session_list = []
    flexilims_session = flz.get_flexilims_session(project)
    for mouse in mouse_list:
        sessions_mouse = flz.get_children(
            parent_name=mouse,
            children_datatype="session",
            flexilims_session=flexilims_session,
        ).name.values.tolist()
        session_list.append(sessions_mouse)

    # exclude any sessions from exclude_sessions
    session_list = [session for i in session_list for session in i]
    session_list = [
        session for session in session_list if session not in exclude_sessions
    ]

    return session_list


def get_all_sessions(project, mouse_list, closedloop_only=True, openloop_only=False):
    session_list = []

    # Exclude non-V1 sessions
    exclude_sessions = [
        "PZAH6.4b_S20220428",
        "PZAH6.4b_S20220510",
        "PZAH6.4b_S20220526",
        "PZAG3.4f_S20220422",
        "PZAG3.4f_S20220429",
        "PZAG3.4f_S20220508",
        "PZAG3.4f_S20220509",
        "PZAG3.4f_S20220517",
        "PZAG3.4f_S20220519",
        "PZAH8.2h_S20221208",
        "PZAH8.2h_S20221213",
        "PZAH8.2h_S20221215",
        "PZAH8.2h_S20221216",
        "PZAH10.2d_S20230615",
        "PZAH10.2d_S20230616",
        "PZAH10.2d_S20230626",
        "PZAH10.2f_S20230525",
        "PZAH10.2f_S20230601",
        "PZAH10.2f_S20230614",
        "PZAH10.2f_S20230626",
        "PZAH10.2f_S20230705",
        "PZAH10.2f_S20230801",
        "PZAH10.2f_S20230804",
        "PZAH10.2f_S20230808",
        "PZAH10.2f_S20230811",
    ]

    # Exclude Faulty sessions --> DEBUG!!
    exclude_sessions = exclude_sessions + [
        "PZAH6.4b_S20220523",
        "PZAH8.2h_S20230126",
        "PZAH8.2h_S20230310",
        "PZAH8.2f_S20230210",
        "PZAH8.2f_S20230213",
        "PZAH8.2f_S20230214",
        "PZAH8.2f_S20230323",
        "PZAH10.2d_S20230623",
        "PZAH10.2f_S20230605",
    ]

    # Exclude sessions with no proper settings
    exclude_sessions = exclude_sessions + [
        "PZAH6.4b_S20220401",  # not proper settings
        "PZAH6.4b_S20220411",  # not proper settings
        "PZAH6.4b_S20220513",  # not proper settings
        "PZAH8.2h_S20221208",  # test
        "PZAH8.2h_S20221213",  # 10 depths
        "PZAH8.2h_S20221215",  # test
        "PZAH8.2h_S20221216",  # 10 depths
        "PZAH8.2h_S20230411x1024",  # test for 10x objective
        "PZAH8.2h_S20230411x2048",  # test for 10x objective
        "PZAH8.2h_S20230411",  # test for 10x objective
        "PZAH8.2h_S20230410",  # test for 10x objective
        "PZAH8.2i_S20221208",  # test
        "PZAH8.2i_S20221209",  # 10 depths
        "PZAH8.2i_S20221213",  # test
        "PZAH8.2i_S20221215",  # 10 depths
        "PZAH8.2f_S20221206",  # test
        "PZAH8.2f_S20221209",  # 10 depths
        "PZAH8.2f_S20221212",  # test
        "PZAH10.2f_S20230911",  # multiple cortical depths
    ]

    # Exclude size control sessions
    exclude_sessions = exclude_sessions + [
        "PZAH10.2d_S20230822",
        "PZAH10.2f_S20230815",
        "PZAH10.2f_S20230907",
    ]

    if (closedloop_only == True) and (openloop_only == True):
        print(
            "WARNING: Both closedloop_only and openloop_only are True. No session is returned."
        )

    if closedloop_only:
        exclude_sessions = exclude_sessions + [
            "PZAH6.4b_S20220519",
            "PZAH6.4b_S20220524",
            "PZAG3.4f_S20220520",
            "PZAG3.4f_S20220523",
            "PZAG3.4f_S20220524",
            "PZAG3.4f_S20220526",
            "PZAG3.4f_S20220527",
            "PZAH8.2h_S20230224",
            "PZAH8.2h_S20230302",
            "PZAH8.2h_S20230303",
            # "PZAH8.2h_S20230310",
            "PZAH8.2i_S20230203",
            "PZAH8.2i_S20230209",
            "PZAH8.2i_S20230216",
            "PZAH8.2i_S20230220",
            "PZAH8.2f_S20230206",
            # "PZAH8.2f_S20230210",
            # "PZAH8.2f_S20230213",
            # "PZAH8.2f_S20230214",
            "PZAH8.2f_S20230223",
            "PZAH10.2d_S20230602",
            "PZAH10.2d_S20230608",
            "PZAH10.2d_S20230613",
            "PZAH10.2d_S20230818",
            "PZAH10.2d_S20230821",
            "PZAH10.2d_S20230920",
            "PZAH10.2d_S20230922",
            "PZAH10.2f_S20230606",
            "PZAH10.2f_S20230609",
            "PZAH10.2f_S20230615",
            "PZAH10.2f_S20230623",
            "PZAH10.2f_S20230627",
            "PZAH10.2f_S20230822",
            "PZAH10.2f_S20230908",
            "PZAH10.2f_S20230924",
        ]
        session_list = session_list + (
            get_sessions(project, mouse_list, exclude_sessions=exclude_sessions)
        )

    elif openloop_only:
        openloop_sessions = [
            "PZAH6.4b_S20220519",
            "PZAH6.4b_S20220524",
            "PZAG3.4f_S20220520",
            "PZAG3.4f_S20220523",
            "PZAG3.4f_S20220524",
            "PZAG3.4f_S20220526",
            "PZAG3.4f_S20220527",
            "PZAH8.2h_S20230224",
            "PZAH8.2h_S20230302",
            "PZAH8.2h_S20230303",
            # "PZAH8.2h_S20230310",
            "PZAH8.2i_S20230203",
            "PZAH8.2i_S20230209",
            "PZAH8.2i_S20230216",
            "PZAH8.2i_S20230220",
            "PZAH8.2f_S20230206",
            # "PZAH8.2f_S20230210",
            # "PZAH8.2f_S20230213",
            # "PZAH8.2f_S20230214",
            "PZAH8.2f_S20230223",
            "PZAH10.2d_S20230602",
            "PZAH10.2d_S20230608",
            "PZAH10.2d_S20230613",
            "PZAH10.2d_S20230818",
            "PZAH10.2d_S20230821",
            "PZAH10.2d_S20230920",
            "PZAH10.2d_S20230922",
            "PZAH10.2f_S20230606",
            "PZAH10.2f_S20230609",
            "PZAH10.2f_S20230615",
            "PZAH10.2f_S20230623",
            "PZAH10.2f_S20230627",
            "PZAH10.2f_S20230822",
            "PZAH10.2f_S20230908",
            "PZAH10.2f_S20230924",
        ]

        session_list = []
        for mouse in mouse_list:
            session_list = session_list + [
                session for session in openloop_sessions if mouse in session
            ]

    elif (closedloop_only == False) and (openloop_only == False):
        session_list = session_list + (
            get_sessions(project, mouse_list, exclude_sessions=exclude_sessions)
        )

    session_list.sort()
    return session_list
