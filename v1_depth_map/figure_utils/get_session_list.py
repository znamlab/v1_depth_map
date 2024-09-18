import flexiznam as flz


def get_sessions(
    flexilims_session,
    exclude_sessions=(),
    exclude_openloop=True,
    exclude_pure_closedloop=False,
    v1_only=True,
    trialnum_min=10,
    mouse_list=None,
):
    """
    Get a list of sessions to include.

    Args:
        flexilims_session (str): flexilims session
        exclude_sessions (list, optional): list of sessions to exclude manually.
            Defaults to ().
        exclude_openloop (bool, optional): only include closedloop sessions. Defaults to
            True. To include all sessions, set to False.
        exclude_pure_closedloop (bool, optional): only include openloop sessions. Defaults to
            False. To include all sessions, set to False.
        v1_only (bool, optional): only include V1 sessions. Defaults to True.
        mouse_list (list, optional): list of mice to include, if None, include all.
            Default to None.

    Returns:
        list: list of sessions to include

    """
    assert not (
        exclude_openloop and exclude_pure_closedloop
    ), "Both closedloop_only and openloop_only cannot be True"
    session_list = []
    if mouse_list is None:
        mouse_list = flz.get_entities("mouse", flexilims_session=flexilims_session)

    # get children is too slow. It's better to get everything and filter
    project_sessions = flz.get_entities("session", flexilims_session=flexilims_session)
    project_recordings = flz.get_entities(
        "recording", flexilims_session=flexilims_session
    )
    for mouse_id in mouse_list["id"].values:
        sessions_mouse = project_sessions[project_sessions.origin_id == mouse_id]
        # exclude any sessions which have an "exclude_reason" on flexilims
        if "exclude_reason" in sessions_mouse.columns:
            if not v1_only:
                sessions_mouse = sessions_mouse[
                    (sessions_mouse["exclude_reason"].isna())
                    | (sessions_mouse["exclude_reason"].str.isspace())
                    | (sessions_mouse["exclude_reason"] == "not V1")
                ]
            else:
                sessions_mouse = sessions_mouse[
                    (
                        (sessions_mouse["exclude_reason"].isna())
                        | (sessions_mouse["exclude_reason"].str.isspace())
                    )
                    & (
                        sessions_mouse["closedloop_trials"] / sessions_mouse["ndepths"]
                        > trialnum_min
                    )
                ]
        if len(sessions_mouse) > 0:
            session_list.append(sessions_mouse.name.values.tolist())
    # exclude any sessions from exclude_sessions
    session_list = [session for i in session_list for session in i]
    session_list = [
        session for session in session_list if session not in exclude_sessions
    ]
    keep_sessions = []
    # filter based on closedloop_only and openloop_only
    for session in session_list:
        sess_id = project_sessions.loc[session].id
        recs = project_recordings[project_recordings.origin_id == sess_id]
        open_loop = (recs["protocol"] == "SpheresPermTubeRewardPlayback").any()
        if exclude_openloop and open_loop:
            continue
        if exclude_pure_closedloop and (not open_loop):
            continue
        keep_sessions.append(session)
    return keep_sessions


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

    if closedloop_only and openloop_only:
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

    elif (not closedloop_only) and (not openloop_only):
        session_list = session_list + (
            get_sessions(project, mouse_list, exclude_sessions=exclude_sessions)
        )

    session_list.sort()
    return session_list