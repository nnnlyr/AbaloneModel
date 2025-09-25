import requests
from bs4 import BeautifulSoup

def print_character(url):
    # Send a GET request to the URL and check if the request was successful (status code 200)
    response = requests.get(url)
    if response.status_code == 200:
        # Parse the HTML content of the page using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first <table> element in the parsed HTML
        table = soup.find('table')

        # If the table is found
        if table:
            # Find all rows in the table
            table_rows = table.find_all('tr')

            # Initialize an empty dictionary to store characters and coordinates
            characters = {}
            # Initialize variables to track the maximum x and y coordinates
            x_max = 0
            y_max = 0

            # Loop through each row in the table
            for row in table_rows:
                # Find all cells in the row
                cells = row.find_all('td')

                # Ensure the row has 3 cells (x-coordinate, character, y-coordinate)
                if len(cells) == 3:
                    try:
                        # Get the x
                        x = int(cells[0].get_text().strip())

                        # Update x_max if the current x is bigger
                        if x > x_max:
                            x_max = x

                        # Get the character
                        character = cells[1].get_text().strip()

                        # Get the x
                        y = int(cells[2].get_text().strip())

                        # Update y_max if the current y is bigger
                        if y > y_max:
                            y_max = y

                        # Use setdefault to store the character at the (x, y) coordinate
                        characters.setdefault((x,y), character)
                    except ValueError:
                        continue

            # Calculate the grid size based on the maximum x and y coordinates found
            grid_x_size = x_max + 1
            grid_y_size = y_max + 1

            # Create a grid with empty spaces, sized based on the max x and y values
            grid = [[" " for _ in range(grid_x_size)] for _ in range(grid_y_size)]

            # Place the characters at their respective (x, y) coordinates in the grid
            for coord, character in characters.items():
                x, y = coord

                # Adjust y to match the grid's coordinate system (flipping y-axis)
                actual_y = grid_y_size - y - 1

                # Place the character in the grid
                grid[actual_y][x] = character

            # Print the grid row by row to display the result
            for row in grid:
                print("".join(row))


if __name__ == '__main__':
    url = ('https://docs.google.com/document/d/e/2PACX-1vQGUck9HIFCyezsrBSnmENk5ieJuYwpt7YHYEzeNJkIb9OSDdx-ov2nRNReKQyey-cwJOoEKUhLmN9z/pub')
    print_character(url)