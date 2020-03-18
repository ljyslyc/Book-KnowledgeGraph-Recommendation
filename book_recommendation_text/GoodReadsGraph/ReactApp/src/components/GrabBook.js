import React, { useState, useEffect } from 'react';
import { Image } from "semantic-ui-react";

export const GrabBook = ()=> {
  const [outputURL, setOutputURL] = useState("");

  useEffect(() => {
   fetch("/novel_novel").then(response =>
    response.json().then(data => {
      setOutputURL(data.image_url);
      })
    );
  }, []);
  const BookImage = () => (<Image src={outputURL} size='small' />)

  console.log("reponse")
  console.log(outputURL)

  return(
    <BookImage/>
  )

}
