import neptune.new as neptune


def config_run():
    run = neptune.init(
            project="UMN-RC-1/Fatchecker",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzOGFiYmZkMS1jY2VmLTQ1YWYtYjJjZC0xMjc0MTI4MzNiZTAifQ==",
        )

    return run