competition_type = 'allplayall'

players = {
    'lzp-bg': Player('leelaz --gtp -w weights/leela-zero-pytorch-bg.txt',
                      startup_gtp_commands=[
                          'time_settings 0 11 1',
                      ]),

    'lzp-sm': Player('leelaz --gtp -w weights/leela-zero-pytorch-sm.txt',
                     startup_gtp_commands=[
                          'time_settings 0 11 1',
                     ]),
}

board_size = 19
komi = 6.5

rounds = 5
competitors = ['lzp-bg', 'lzp-sm']
