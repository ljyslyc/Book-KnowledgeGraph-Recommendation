import React, { useState } from 'react';
import { Form, Input, Button } from 'semantic-ui-react';


export const BookEntry = ()  => {
  const [title, setTitle] = useState(''); //  Empty String
  return (
    <Form>
      <Form.Field>
        <Input
        placeholder="Enter one of your favorite book titles "
        value={title}
        onChange={event => setTitle(event.target.value)}
        />
      </Form.Field>

      <Form.Field>
        <Button onClick= {async () => {
          const book = {title};
          const response = await fetch("/input_book", {
            method: "POST",
            headers: {
              "Content_Type": "application/json"
            },
            body:
              JSON.stringify(book)
            })

          if (response.ok) {
            console.log("Response Worked! ");
            console.log(JSON.stringify(response.url));
            console.log(response);
            setTitle("We found your favorite book!")
            console.log(response);
          }
          else {
            console.log("Title not found")
            setTitle("We did not find this title. Please try again!")
          }

        }}>

        Add</Button>
      </Form.Field>
    </Form>
  );
};
