# Card games 
Repository for years project. 

Project website http://www.st.fmph.uniba.sk/~kassak7/rp_2.html

## How to run and test
First you need to install `docker` (https://www.docker.com/). 

To run tests type:
```commandline
docker compose up testing
```
To run the game:
```commandline
docker compose run --rm console_pharaoh
```
To customize game configuration edit `./src/config/config_pharaoh.yml`.

To run analysis:
```commandline
docker compose up analysis
```